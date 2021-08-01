#include "data/vocab_base.h"

#ifdef USE_SENTENCEPIECE
#include "sentencepiece/src/sentencepiece_processor.h"
#include "sentencepiece/src/sentencepiece_trainer.h"
#endif

#include "common/filesystem.h"
#include "common/logging.h"
#include "common/options.h"

namespace marian {

#ifdef USE_SENTENCEPIECE

#define DEBUG_ZCODE_VOCAB // TODO (amralaa)

// Wrapper around https://github.com/google/sentencepiece
class ZCodeVocab : public IVocab {
private:
  typedef std::map<std::string, Word> Str2Id;
  Str2Id langTokenToId_;
  size_t maxTokenId_{0};

  // Actual SentencePiece processor object
  UPtr<sentencepiece::SentencePieceProcessor> spm_;

  // Allowed suffixes for SentencePiece model
  std::vector<std::string> suffixes_ = {".spm"};

  Ptr<Options> options_;

  std::uniform_int_distribution<int> randInt_;  // from 0 to INT_MAX

  // Keeps sentences segmented into subword units
  bool keepEncoded_;

  size_t maxLength_;

  bool addPreLangEos_;

  size_t spmSpecialTokensCount_{3};
  size_t fsDictSpecialTokensCount_{4};

  size_t fsDictEosId_{2};
  size_t fsDictUnkId_{3};

  void populateLangTokenIds() {
    langTokenToId_.clear();

    auto langs = utils::split(options_->get<std::string>("lang-tokens"), ",");
    size_t currentIdx = options_->get<int>("lang-tokens-start-index");
    for(const auto& lang : langs) {
      // TODO (amralaa): Is there any method like std::format("__{}__", lang)?
      std::string langToken = "__" + lang + "__";
      langTokenToId_[langToken] = Word::fromWordIndex(currentIdx);
      ++currentIdx;
    }

    maxTokenId_ = currentIdx - 1;
  }

  size_t spmToFs(size_t spmId) const {
    size_t fsId;
    if(spmId >= spmSpecialTokensCount_) {
      fsId = spmId - spmSpecialTokensCount_ + fsDictSpecialTokensCount_;
    } else if(spmId == spm_->eos_id()) {
      fsId = fsDictEosId_;
    } else if(spmId == spm_->unk_id()) {
      fsId = fsDictUnkId_;
    } else {
      ABORT("Unsupported spm Id: {}", spmId);
    }
    return fsId;
  }

  size_t fsToSpm(size_t fsId) const {
    size_t spmId;
    if(fsId >= fsDictSpecialTokensCount_) {
      spmId = fsId - fsDictSpecialTokensCount_ + spmSpecialTokensCount_;
      ABORT_IF(spmId >= spm_->GetPieceSize(), "spmId >= spm_->GetPieceSize()");
    } else if(fsId == fsDictEosId_) {
      spmId = spm_->eos_id();
    } else if(fsId == fsDictUnkId_) {
      spmId = spm_->unk_id();
    } else {
      ABORT("Unsupported fairseq Id: {}", fsId);
    }
    return spmId;
  }

public:
  ZCodeVocab(Ptr<Options> options, size_t batchIndex)
      : options_(options),
        keepEncoded_(options->get<bool>("no-spm-decode", false)),
        maxLength_(options->get<int>("max-length")),
        addPreLangEos_(true),
        spmSpecialTokensCount_(options->get<int>("spm-special-tokens-count")),
        fsDictSpecialTokensCount_(options->get<int>("fs-special-tokens-count")),
        fsDictEosId_(options->get<int>("eos-index")),
        fsDictUnkId_(options->get<int>("unk-index")) {
    populateLangTokenIds();
  }

  virtual const std::string& canonicalExtension() const override { return suffixes_[0]; }
  virtual const std::vector<std::string>& suffixes() const override { return suffixes_; }

  virtual std::string suffix() { return suffixes_[0]; };

  virtual std::string type() const override { return "ZCodeVocab"; }

  virtual Word getEosId() const override { return Word::fromWordIndex(fsDictEosId_); }
  virtual Word getUnkId() const override { return Word::fromWordIndex(fsDictUnkId_); }

  void create(const std::string& vocabPath,
              const std::vector<std::string>& trainPaths,
              size_t maxSize) override {
    ABORT("[ZcodeVocab] Creation of ZCode vocabulary is not supported");
  }

  void createFake() override { ABORT("[ZcodeVocab] Fake ZCode vocabulary is not supported"); }

  Word operator[](const std::string& token) const override {
    auto spmId = spm_->PieceToId(token);
    auto fsId = spmToFs(spmId);
    return Word::fromWordIndex(fsId);
  }

  const std::string& operator[](Word id) const override {
    auto fsId = id.toWordIndex();
    ABORT_IF(fsId >= size(), "Unknown word id: {}", fsId);
    auto spmId = fsToSpm(fsId);
    return spm_->IdToPiece(spmId);
  }

  Words encode(const std::string& line, bool addEOS, bool inference) const override {
    ABORT_IF(!inference, "ZCodeVocab is supported only for inference.");

    LOG(debug, "Input line: {}", line);  // TODO (amralaa)

    size_t langTokenLength = 6;
    // TODO (amralaa): How to get target language?
    std::string langToken = line.substr(line.size() - langTokenLength);
    std::string content = line.substr(0, line.size() - langTokenLength - 1);

    std::vector<int> spmIds;
    spm_->Encode(content, &spmIds);

#ifdef DEBUG_ZCODE_VOCAB
    std::vector<int> fsIds;
    for(auto& spmId : spmIds) {
      LOG(debug, "SPMId: {}", std::to_string(spmId));
      ABORT_IF(spmId < spmSpecialTokensCount_, "Unexpected token id {}", std::to_string(spmId));
      auto fsId = spmToFs(spmId);
      LOG(debug, "FSId: {}", std::to_string(fsId));
      fsIds.push_back(fsId);
    }

    // Split into SentencePiece pieces
    std::vector<std::string> pieces;
    spm_->Encode(content, &pieces);
    for(auto& piece : pieces) {
      LOG(debug, "Piece: {}", piece);
    }

    ABORT_IF(pieces.size() != spmIds.size(), "Mismatching size");
#endif

    size_t maxTokens = maxLength_;
    size_t appendedTokensCount = 1 + addPreLangEos_ + addEOS;
    ABORT_IF(maxTokens <= appendedTokensCount, "maxTokens <= appendedTokensCount");
    size_t maxContentTokens = maxTokens - appendedTokensCount;  // 3 appended tokens: EOS LANG EOS
    size_t contentTokensCount = std::min(maxContentTokens, spmIds.size());

#ifdef DEBUG_ZCODE_VOCAB
    std::vector<std::string> lineTokens;
    lineTokens.reserve(contentTokensCount + appendedTokensCount);
    for(int i = 0; i < contentTokensCount; ++i) {
      lineTokens.push_back(pieces[i]);
    }

    for(auto& token : lineTokens) {
      LOG(debug, "Line token: {}", token);
    }
#endif

    // Truncate tokens
    Words words;
    words.reserve(contentTokensCount + appendedTokensCount);
    for(int i = 0; i < contentTokensCount; ++i) {
      auto spmId = spmIds[i];
      auto fsId = spmToFs(spmId);
      words.push_back(Word::fromWordIndex(fsId));
    }

    if(addPreLangEos_)
      words.push_back(getEosId());

    Word langTokenId = langTokenToId_.at(langToken);
    words.push_back(langTokenId);

    if(addEOS)
      words.push_back(getEosId());

#ifdef DEBUG_ZCODE_VOCAB
    for(auto& word : words) {
      LOG(debug, "Final words: {}", word.toString());
    }
#endif

    ABORT_IF(words.size() > maxLength_, "words.size() > maxLength_");
    return words;
  }

  std::string decode(const Words& sentence, bool /*ignoreEOS*/) const override {
    std::string line;
    if(keepEncoded_) {  // i.e. keep the sentence segmented into subword units
      for(const Word& id : sentence)
        line += (*this)[id] + " ";
      line.pop_back();  // trim the trailing whitespace
    } else {
      // convert vector of Word to vector of int
      std::vector<int> spmSentence;
      spmSentence.reserve(sentence.size());
      for(auto&& word : sentence) {
        auto fsId = word.toWordIndex();
        auto spmId = fsToSpm(fsId);
        spmSentence.push_back(spmId);
      }
      spm_->Decode(spmSentence, &line);
    }
    return line;
  }

  std::string surfaceForm(const Words& sentence) const override {
    // with SentencePiece, decoded form and surface form are identical
    return decode(sentence, /*ignoreEOS=*/true);
  }

  size_t size() const override { return maxTokenId_ + 1; }

  size_t load(const std::string& vocabPath, size_t /*maxSize*/) override {
    LOG(info, "[data] Loading SentencePiece vocabulary from file {}", vocabPath);

    ABORT_IF(!filesystem::exists(vocabPath),
             "SentencePiece vocabulary file {} does not exist",
             vocabPath);

    spm_.reset(new sentencepiece::SentencePieceProcessor());
    const auto status = spm_->Load(vocabPath);

    ABORT_IF(!status.ok(), "SentencePiece vocabulary error: {}", status.ToString());

    return spm_->GetPieceSize();
  }

  std::string toUpper(const std::string& line) const override { return utils::utf8ToUpper(line); }
  std::string toEnglishTitleCase(const std::string& line) const override {
    return utils::toEnglishTitleCase(line);
  }
};
#endif  // USE_SENTENCEPIECE

Ptr<IVocab> createZCodeVocab(const std::string& vocabPath,
                             Ptr<Options> options,
                             size_t batchIndex) {
#ifdef USE_SENTENCEPIECE
  return New<ZCodeVocab>(options, batchIndex);
#else
  batchIndex;
  options;
  ABORT(
      "Support for SentencePiece is not compiled into Marian. "
      "Try to recompile after `cmake .. -DUSE_SENTENCEPIECE=on [...]`",
      vocabPath);
#endif
  return nullptr;
}

}  // namespace marian
