#pragma once

#include "data/iterator_facade.h"
#include "data/corpus.h"

namespace marian {
namespace data {

class TextInput;

class TextIterator : public IteratorFacade<TextIterator, SentenceTuple const> {
public:
  TextIterator();
  explicit TextIterator(TextInput& corpus);

private:
  void increment() override;

  bool equal(TextIterator const& other) const override;

  const SentenceTuple& dereference() const override;

  TextInput* corpus_;

  long long int pos_;
  SentenceTuple tup_;
};

class TextInput : public DatasetBase<SentenceTuple, TextIterator, CorpusBatch> {
private:
  std::vector<UPtr<std::istringstream>> files_;
  std::vector<Ptr<Vocab>> vocabs_;

  size_t pos_{0};

  size_t maxLength_{0};
  bool maxLengthCrop_{false};

public:
  typedef SentenceTuple Sample;

  TextInput(std::vector<std::string> inputs, std::vector<Ptr<Vocab>> vocabs, Ptr<Options> options);
  virtual ~TextInput() {}

  Sample next() override;

  void shuffle() override {}
  void reset() override {}

  iterator begin() override { return iterator(*this); }
  iterator end() override { return iterator(); }

  // TODO: There are half dozen functions called toBatch(), which are very
  // similar. Factor them.
  batch_ptr toBatch(const std::vector<Sample>& batchVector) override {
    size_t batchSize = batchVector.size();

    std::vector<size_t> sentenceIds;

    std::vector<int> maxDims;
    for(auto& ex : batchVector) {
      if(maxDims.size() < ex.size())
        maxDims.resize(ex.size(), 0);
      for(size_t i = 0; i < ex.size(); ++i) {
        if(ex[i].size() > (size_t)maxDims[i])
          maxDims[i] = (int)ex[i].size();
      }
      sentenceIds.push_back(ex.getId());
    }

    std::vector<Ptr<SubBatch>> subBatches;
    for(size_t j = 0; j < maxDims.size(); ++j) {
      subBatches.emplace_back(New<SubBatch>(batchSize, maxDims[j], vocabs_[j]));
    }

    std::vector<size_t> words(maxDims.size(), 0);
#define HAS_PRE_LANG_EOS
#ifndef HAS_PRE_LANG_EOS
    for(size_t i = 0; i < batchSize; ++i) {
      for(size_t j = 0; j < maxDims.size(); ++j) {
        for(size_t k = 0; k < batchVector[i][j].size(); ++k) {
          subBatches[j]->data()[k * batchSize + i] = batchVector[i][j][k];
          subBatches[j]->mask()[k * batchSize + i] = 1.f;
          words[j]++;
        }
      }
    }
#else
    for(size_t i = 0; i < batchSize; ++i) {                    // loop over batch entries
      for(size_t j = 0; j < maxDims.size(); ++j) {             // loop over streams // async branch
        auto& subBatch = subBatches[j];
        for(size_t k = 0; k < batchVector[i][j].size(); ++k) { // loop over word positions
          if(k < batchVector[i][j].size() - 3) {
            subBatch->data()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = batchVector[i][j][k];
            subBatch->mask()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = 1.f;
          } else if (k == batchVector[i][j].size() - 3) {
            // Add EOS with no masking 
            subBatch->data()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = batchVector[i][j][batchVector[i][j].size() - 1];
            subBatch->mask()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = 1.f;
          } else if (k == batchVector[i][j].size() - 2) {
            // Add the lang id
            subBatch->data()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = batchVector[i][j][k];
            subBatch->mask()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = 1.f;
          } else if (k == batchVector[i][j].size() - 1) {
            // Add EOS with masking
            subBatch->data()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = batchVector[i][j][k];
            subBatch->mask()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/] = 0.f;
          }

          LOG(info,
            "Hossam debug indices word id: {} loc: {} mask: {}",
            batchVector[i][j][k].toWordIndex(),
            subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k),
            subBatch->data()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/].toWordIndex(),
            subBatch->mask()[subBatch->locate(/*batchIdx=*/i, /*wordPos=*/k)/*k * batchSize + i*/]);

          words[j]++;
        }
      }
    }
#endif

    for(size_t j = 0; j < maxDims.size(); ++j)
      subBatches[j]->setWords(words[j]);

    auto batch = batch_ptr(new batch_type(subBatches));
    batch->setSentenceIds(sentenceIds);

    return batch;
  }

  void prepare() override {}
};
}  // namespace data
}  // namespace marian
