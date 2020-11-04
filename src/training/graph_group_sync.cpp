#include "training/graph_group_sync.h"

#include <math.h>

namespace marian {

SyncGraphGroup::SyncGraphGroup(Ptr<Options> options, Ptr<IMPIWrapper> mpi)
    : GraphGroup(options, Config::getDevices(options, mpi->myMPIRank(), mpi->numMPIProcesses())),
      delay_{options_->get<double>("optimizer-delay")}, mpi_(mpi) { // @TODO: rename delay_ to something else; delay means delayed updated, not accumulation

  GraphGroup::initGraphs();

  for(auto graph : graphs_) {
    graph; // avoid unused variable error
    optimizerShards_.push_back(Optimizer(options_));
    models_.push_back(models::createCriterionFunctionFromOptions(options_, models::usage::training));
  }

  // Note: We may well end up with only one MPI process or only one graph per worker.
  // This part of the code will not special-case any of this here.
  // Rather, it is assumed that the communicator knows to reduce unnecessary transfers to no-ops.
  comm_ = createCommunicator(graphs_, /*noNccl=*/options_->get<bool>("no-nccl", false), /*mpi=*/mpi_);

  auto formattedDeviceType = utils::utf8ToUpper(devices_.front().typeAsString()) + "s";
  if (mpi_->numMPIProcesses() > 1)
    LOG(info, "[training] Using {} {}, distributed over {} MPI processes", mpi_->numMPIProcesses() * devices_.size(), formattedDeviceType, mpi_->numMPIProcesses());
  else
    LOG(info, "[training] Using {} {}", devices_.size(), formattedDeviceType);
}

void SyncGraphGroup::setScheduler(Ptr<Scheduler> scheduler) /*override*/ {
  validate();
  scheduler_ = scheduler;
  scheduler_->registerTrainingObserver(scheduler_);

  // optimizer has to be registered last to see changes of learning rate
  for(auto opt : optimizerShards_)
    scheduler_->registerTrainingObserver(opt);
}

void SyncGraphGroup::initialize(const Ptr<data::Batch>& exampleBatch) {
  // Initialize graphs with random weights in one forward step
  // Also allocate and clear the gradients
  comm_->foreach([&](size_t i, size_t /*begin*/, size_t /*end*/) {
    models_[i]->build(graphs_[i], exampleBatch);
    graphs_[i]->forward();
    graphs_[i]->params()->allocateBackward();
    graphs_[i]->params()->set_zero_adjoint();
    return true; // dummy success
  });

  // Copy weights from 0-th graph to all other graphs
  // to have equal weights across devices
  comm_->foreach([&](size_t i, size_t /*begin*/, size_t /*end*/) {
    if (i > 0)
      graphs_[i]->params()->vals()->copyFrom(graphs_[0]->params()->vals());
    return true; // dummy success
  });
  
  // initialize model quantization
  if (options_->get<size_t>("quantize-bits") > 0) {
    for (int idx = 0; idx < graphs_.size(); idx++)
      quantizers_.push_back(New<ModelQuantizer>(options_));
    
    comm_->foreach([&](size_t idx, size_t /*begin*/, size_t /*end*/) { quantizers_[idx]->quantize(graphs_[idx]); return true; });
  }

  // We compute the readerMultiplier in collectStats(...) and the updateMultiplier_ here
  // as collectStats maybe called for a different instance of this object and fields would not
  // survive destruction.
  double multiplier = devices_.size() * mpi_->numMPIProcesses() * delay_;
  bool isDynamic = scheduler_->isDynamicMBSizeScaling();
  updateMultiplier_ = isDynamic ? multiplier : 1.; // multiplier applied later in update()
}

Ptr<data::BatchStats> SyncGraphGroup::collectStats(const std::vector<Ptr<Vocab>>& vocabs) {
  // This function determines the granularity in which the reader provides data.
  // If no mini-batch-fit, then user provides a constant number. It reads that much. We won't get into this function.
  
  // If dynamic MB scaling, then we want fine-grained minibatches of the size of one GPU.
  // If not, we prefer a single large batch that can be split into equal-size parts over GPUs,
  // so that we have perfect load balancing and read precisely as much as we need (no waste).
  double multiplier = devices_.size() * mpi_->numMPIProcesses() * delay_;
  bool isDynamic = scheduler_->isDynamicMBSizeScaling();
  double readerMultiplier = isDynamic ? 1. : multiplier; // multiplier applied already by reader
  return GraphGroup::collectStats(graphs_[0], models_[0], vocabs, readerMultiplier);
}

// helper for MB scaling: quantize the ratio with a given error margin
static double roundUpRatio(double ratio) {
  if (ratio == 0)
    return ratio;
  // find largest power of two that fits into ratio
  double p = 1;
  while (p*2 < ratio)
    p *= 2;
  // round up to nearest multiple of a largest power of 2 where relative error is within margin
  // 25% error margin seems acceptable:
  //  - using a 25% larger MB size should not break convergence
  //  - @TODO: not using the first 25% of the next block is OK since those are dominated by data exchange
  double maxError = 0.25;
  while (p >= 1) {
    double proposedRatio = ceil(ratio / p) * p;
    double error = (proposedRatio - ratio) / ratio;
    if (fabs(error) <= maxError)
      return proposedRatio;
    p /= 2;
  }
  return ratio;
}

// helper routine that handles accumulation and load-balancing of sub-batches to fill all devices
// It adds 'newBatch' to 'pendingBatches_', and if sufficient batches have been queued, then
// returns 'pendingBatches_' in 'subBatches' and resets it. If not, it returns false.
bool SyncGraphGroup::tryGetSubBatches(Ptr<data::Batch> newBatch,
    std::vector<Ptr<data::Batch>>& subBatches, size_t& numReadBatches) {
  // The reader delivers in chunks of these sizes, according to case:
  //  - no dynamic MB-size scaling:
  //     - reader batch size = update batch size, with...
  //     - mini-batch-fit:
  //        - update batch size = what fits into all GPUs, times decay_ to allow experimenting with fractional sizes
  //     - no mini-batch-fit:
  //        - update batch size = user-specified size (user guarantees that it fits if distributed over delay_ GPUs)
  //  - dynamic MB-size scaling:
  //     - update batch size = aggregate reader batch size * (dynamic progress-based ratio * reference adjustment), with...
  //     - mini-batch-fit:
  //        - aggregate reader batch size = equal to what fits into one GPU * warpSize * delay_
  //     - no mini-batch-fit:
  //        - aggregate reader batch size = user-specified size (user guarantees that it fits if distributed over delay_ GPUs)
  //     - reference adjustment =
  //        - reference batch size specified: (reference batch size / typical aggregate reader batch size)
  //        - no ref size specified: 1

  size_t warpSize = devices_.size() * mpi_->numMPIProcesses(); // warp := set of batches processed concurrently across GPus and workers

  // if not dynamic then return the big batch, but first split it over GPUs as it may be too large
  if (!scheduler_->isDynamicMBSizeScaling()) {
    // If mini-batch-fit, then the read batch is (devices_.size() * mpi_->numMPIProcesses() * delay_)
    // times what fits one GPU. If not mini-batch-fit, it is whatever the user has specified, which
    // is the user's responsibility to guarantee that it fits into 'delay_' warps.
    // Distribute evenly over all GPUs we have, using multiple warps if needed.
    size_t numWarps = (size_t)ceil(delay_);
    subBatches = newBatch->split(numWarps * warpSize); // @TODO might be eaiser to mb-scaling here if mb-words not given?
    numReadBatches = 1;
    return true;
  }
  LOG_ONCE(info, "[training] Dynamic mini-batch scaling enabled");

  // if dynamic and mini-batch-fit, then we get batches in the size of what fits into one GPU
  pendingBatches_.push_back(newBatch);

  // what ratio (how many batches in reader's batch size) do we want, based on current training progress schedule?
  double ratio = scheduler_->getDynamicMBSizeMultiplier();

  // relative to what base? (what does ratio == 1 mean)
  // updateMultiplier_ is only used if we do mini-batch warmup and did not provide mini-batch-words. Otherwise it gets cancelled out.
  ratio *= updateMultiplier_; // if mini-batch-fit, this is = warpSize * delay_, otherwise 1

  // If a reference is given, then at progress == mbWarmup.n (ratio=1), we would like to have refBatchLabels instead of whichever
  // the actual batch size is. Since we cannot know the future actual batch sizes that will be delivered
  // by the reader, we approximate them with (typicalTrgBatchWords * updateMultiplier), and scale ratio accordingly.
  auto refBatchLabels = options_->get<size_t>("mini-batch-words");
  if (refBatchLabels != 0) {
    LOG_ONCE(info, "[scheduler] Scaling to {} reference labels, using actual-batch-word estimate of {}", refBatchLabels, GraphGroup::getTypicalTrgBatchWords());
    ABORT_IF(GraphGroup::getTypicalTrgBatchWords() == 0, "Dynamic scaling with words target requires MB size to be known in words"); // happens if MB size is specified in sentences

    GraphGroup::updateAverageTrgBatchWords(newBatch->wordsTrg());
    ratio *= (double)refBatchLabels / (GraphGroup::getTypicalTrgBatchWords() * updateMultiplier_); // cancellation of updateMultiplier_
  }

  // round up to full batches if within a certain error margin  --@BUGBUG: Not invariant w.r.t. GPU size, as ratio is relative to what fits into 1 GPU
  if(GraphGroup::mbRoundUp_) // true by default
    ratio = roundUpRatio(ratio);

  if (pendingBatches_.size() < ratio)
    return false; // not enough data yet

  // now we have enough to fill at least 'ratio' batches
  // @BUGBUG: We do not handle the case that fixed MB size * ratio exceeds GPU memory (we'd need to split that).

  numReadBatches = pendingBatches_.size(); // remember original batch-counter increment from reader (which is not always the same as subBatches.size() in the end)

  // in fact, we got too much, so make up for it by shortening all batches to accurately reflect desired ratio
  // e.g. ratio = 3.3 for 4 batches -> Reduce each by 3.3/4
  // Alternatively, we could just shorten the last 'warp', but that would not be invariant to warp size.
  for (auto& batch : pendingBatches_) {
    auto reducedBatchSize = (size_t)ceil((double)batch->size() * ratio / (double)pendingBatches_.size());
    size_t minSize = 1;
    if (pendingBatches_.size() == 1) { // enforce a minimum (only needed/correct if still in first batch)
      size_t minTrgWords = 256;        // don't go below this number of target words, as it seems excessive  --@TODO: parameterize?
      minSize = 1 + (minTrgWords * batch->size() - 1) / batch->wordsTrg(); // approximately convert minTrgWords into a #sentences
    }
    reducedBatchSize = std::max(reducedBatchSize, minSize);
    if (reducedBatchSize < batch->size())
      batch = batch->split(/*numSubBatches=*/1, reducedBatchSize).front();
  }

  // load-balance: distribute the last numWarps-group's batches over GPUs
  // This is tricky since batches do not have the same length, therefore we can only split, but not merge.
  auto numWarps = (pendingBatches_.size() - 1) / warpSize + 1; // = ceil(#buffers / (#GPUs * #workers))
  auto availableDevices = numWarps * warpSize; // we will run this many GPUs: better use them all
  if (pendingBatches_.size() < availableDevices) {
    // last warp does not use all available GPUs: try to re-balance
    auto fullWarpsBatches = (numWarps - 1) * warpSize; // number of batches in all but the last warp. Those warps that are fully used.
    auto lastWarpSize = pendingBatches_.size() - fullWarpsBatches; // the last warp is possibly not fully used
    //LOG(info, "attempting to redistribute last {} batches over {} devices", lastWarpSize, warpSize);
    auto splitInto = warpSize / lastWarpSize;
    if (splitInto > 1) { // unfortunately we can only split in integer ratios
      // split each of last numWarps's batches into 'splitInto' batches
      // pop them first
      std::vector<Ptr<data::Batch>> batchesToSplit;
      while (pendingBatches_.size() > fullWarpsBatches) {
        batchesToSplit.push_back(pendingBatches_.back());
        pendingBatches_.pop_back();
      }
      // now split them and push them back
      for (auto& batchToSplit : batchesToSplit) {
        //LOG(info, "{}-way splitting batchToSplit with size {}", splitInto, batchToSplit->size());
        auto splitBatches = batchToSplit->split(splitInto);
        for (auto& splitBatch : splitBatches) {
          //LOG(info, " -> getting batchToSplit with size {}", splitBatch->size());
          pendingBatches_.push_back(splitBatch);
        }
      }
    }
    ABORT_IF(pendingBatches_.size() > availableDevices, "somehow split into too many batches??");
  }
  subBatches = std::move(pendingBatches_);

  // Order by decreasing batch width to better pack computation in case of delayed updates
  if(subBatches.size() > 1)
    std::sort(subBatches.begin(), subBatches.end(), 
              [](Ptr<data::Batch> a, Ptr<data::Batch> b) { return a->widthTrg() > b->widthTrg(); });

  return true;
}

void SyncGraphGroup::update(Ptr<data::Batch> newBatch) /*override*/ {
  validate();

  std::vector<Ptr<data::Batch>> subBatches;
  size_t numReadBatches; // actual #batches delivered by reader, for restoring from checkpoint   --@TODO: reader should checkpoint itself; should not go via the scheduler
  bool gotSubBatches = tryGetSubBatches(newBatch, subBatches, numReadBatches);

  // not enough data yet: return right away
  if (!gotSubBatches)
    return;

  update(subBatches, numReadBatches);
}

void SyncGraphGroup::update(std::vector<Ptr<data::Batch>> subBatches, size_t numReadBatches) {
  size_t batchSize = 0;
  size_t batchTrgWords = 0;
  for (const auto& batch : subBatches) {
    batchSize     += batch->size();
    batchTrgWords += batch->wordsTrg();
  }

  std::sort(subBatches.begin(), subBatches.end(),
            [](Ptr<data::Batch> a, Ptr<data::Batch> b) { return a->wordsTrg() > b->wordsTrg(); });

  // Helper to access the subBatches array
  auto getSubBatch = [&](size_t warp, size_t localDeviceIndex, size_t rank) -> Ptr<data::Batch> {
    // Warp should be slowest changing dimension. If subBatches are sorted by
    // length, then grouping sentences of similar length into the same delay step can
    // reduce unnecessary time spent in padding.
    auto index = (warp * mpi_->numMPIProcesses() + rank) * devices_.size() + localDeviceIndex;
    if (index < subBatches.size())
      return subBatches[index];
    else
      return nullptr; // null if we reached beyond the end
  };

  // Upon very first execution, reset everything
  if(first_) {
    LOG(info, "[training] Batches are processed as {} process(es) x {} devices/process",
        mpi_->numMPIProcesses(), devices_.size());
    initialize(subBatches.front());
    first_ = false;
  }

  // Compute gradients
  // This happens in multiple steps in case of delay > 1.
  std::vector<StaticLoss> localDeviceLosses(devices_.size()); // [local device index] aggregate cost for each local device
  comm_->foreach([&](size_t localDeviceIndex, size_t /*begin*/, size_t /*end*/) { // parallel across devices. Aggregate for warp > 1.
    auto graph = graphs_[localDeviceIndex];
    // reset gradient  --presently done outside
    //graph->params()->allocateBackward();
    //graph->params()->set_zero_adjoint();
    // This happens in multiple steps if there are more subbatches than devices.
    for (size_t warp = 0; ; warp++) {
      // Execute single forward/backward step
      auto subBatch = getSubBatch(warp, localDeviceIndex, mpi_->myMPIRank());
      if (!subBatch)
        break;

      { // let loss go out of scope, frees memory
        auto rationalLoss = models_[localDeviceIndex]->build(graph, subBatch);
        if(costScaleFactor_ != 1.f)
          rationalLoss->loss() * costScaleFactor_;
        graph->forward();

        localDeviceLosses[localDeviceIndex] += *rationalLoss;
      }

      graph->backward(/*zero=*/false); // (gradients are reset before we get here)
    }

#if 1 // experimental and should eventually be somewhere else
    // Handle local gradient explosion but only clip to largest possible value
    // given number of GPUs and type. Should clip rarely. Also clips inf
    // We do another clipping/rescaling after summation.
    auto gradType = graph->params()->grads()->type();
    if(sizeOf(gradType) < sizeOf(Type::float32)) {
      using namespace functional;
      size_t numGpus = mpi_->numMPIProcesses() * devices_.size();
      float clipValue = NumericLimits<float>(gradType).max / (float)numGpus;
      Element(_1 = clip(_1, clipValue), graph->params()->grads());
    }
#endif

    return true; // dummy success
  });

  // At this point, each device on each MPI process has a gradient aggregated over a subset of the sub-batches.
  // check for Nan or Inf in all summed up shards
  comm_->scatterReduceAndResetGrads();      // reduce gradients across all devices (globally) into shards
  
  bool checkGradient = costScale_ || dynamicGradientScaling_ || checkGradientNan_;

  // Wrapping member function
  auto checkNanOrNorm = [&](size_t i, size_t begin, size_t end) {
    return GraphGroup::checkNanOrNorm(i, begin, end);
  };

  float gradNorm = checkGradient ? comm_->foreach(checkNanOrNorm, accNanOrNorm, 0.f) : 0.f; // @TODO: does this work with MPI?
  bool saneGradient = isFinite(gradNorm);

  if(saneGradient) {
    // actual model update
    auto updateTrgWords = batchTrgWords; // total number of labels across all GPUs and nodes
    float gradientNormalizer = GraphGroup::computeNormalizationFactor(gradNorm, updateTrgWords);

    // Update parameter shard with gradient shard
    auto update = [&](size_t i, size_t begin, size_t end) -> float {
      auto curGrad = graphs_[i]->params()->grads()->subtensor(begin, end-begin);
      auto curParam = graphs_[i]->params()->vals()->subtensor(begin, end-begin);

      float l2norm = optimizerShards_[i]->update(curParam, curGrad, updateTrgWords, gradientNormalizer);
      curGrad->set(0.f); // @TODO: all the different places where gradients get reset are confusing
      return l2norm; // return partial norm
    };

    // Overwrite gradNorm with new value from normalized gradient
    gradNorm = comm_->foreach(update, accNanOrNorm, 0.f); // per-shard model-update

    if(!options_->get<bool>("normalize-gradient"))
      gradNorm /= updateTrgWords; // normalize for logging

    comm_->allGatherParams(); // distribute param value shards back

    // Re-add the error residual from previous quantization,
    // then re-quantize the model back and update the error residual
    if (options_->get<size_t>("quantize-bits") > 0)
      comm_->foreach([&](size_t idx, size_t /*begin*/, size_t /*end*/) { quantizers_[idx]->quantize(graphs_[idx]); return true; });

  } else {
    LOG(debug, "Seen NaN in gradient, skipping update, resetting gradient");

    // Reset gradient shard when no update was done
    auto reset = [&](size_t i, size_t begin, size_t end) {
      auto curGrad = graphs_[i]->params()->grads()->subtensor(begin, end-begin);
      curGrad->set(0.f); // @TODO: all the different places where gradients get reset are confusing
      return true; // dummy success
    };

    gradNorm = 0.f;
    comm_->foreach(reset);   // per-shard model-update
    GraphGroup::decreaseCostScaleFactor();
  }

  // cost across all local devices (scheduler will aggregate cross-process)
  StaticLoss localLoss = std::accumulate(localDeviceLosses.begin(), localDeviceLosses.end(), StaticLoss());

  if(scheduler_) {
    // track and log localLoss
    scheduler_->update(localLoss, numReadBatches, batchSize, batchTrgWords, gradNorm, mpi_);

    // save intermediate model (and optimizer state) to file
    if(scheduler_->saving())
      save();

    // process valid data set
    // This may save a model as well.
    if(scheduler_->validating()) {
      if(isMainProcess()) {
        swapWithSmoothed(graphs_, optimizerShards_, [this]() { comm_->allGatherParams(); });
        scheduler_->validate(graphs_);
        swapWithOriginal(graphs_, optimizerShards_, [this]() { comm_->allGatherParams(); });
      }
    }
  }

  if(saneGradient)
    GraphGroup::increaseCostScaleFactor();
}

void SyncGraphGroup::load() /*override*/ {
  validate();

  auto scatterFn = [&](const io::Item& optimizerState, const OptimizerBase::ScatterStateSetFunc& setShardFn) {
    comm_->scatterState(optimizerState, setShardFn);
  };

  // This function loads the main parameters in the graphs.
  GraphGroup::load(scatterFn);
}

void SyncGraphGroup::save(bool isFinal) /*override*/ {
  auto distParams = [this]() {
    comm_->allGatherParams();
  };

  auto gatherOpt  = [&](const OptimizerBase::GatherStateGetFunc& getShardFn) {
    return comm_->gatherState(getShardFn);
  };

  GraphGroup::save(isFinal, distParams, gatherOpt, isMainProcess());
}

void SyncGraphGroup::finalize() /*override*/ {
  validate();
  Base::finalize();
}

}  // namespace marian
