using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;
using System.IO;
using System.Reflection;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/00-orientation.md
// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/03-bpe-tokenizer.md

namespace CamemBERT
{
    internal class Class2
    {
        public void test1()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");

            IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!");
            Console.WriteLine($"Token count: {ids.Count}");     // Token count: 4
            Console.WriteLine($"Token IDs: [{string.Join(", ", ids)}]");  // Token IDs: [13225, 11, 2375, 0]

            string decoded = tokenizer.Decode(ids)!;
            Console.WriteLine($"Decoded: {decoded}");            // Decoded: Hello, world!
        }

        


        /*
Normalizer 	What It Does 	Used By 
BertNormalizer 	Lowercase, strip accents, clean control chars, handle Chinese chars 	BERT
LowerCaseNormalizer 	Lowercases all text 	Custom
UpperCaseNormalizer 	Uppercases all text 	Custom
SentencePieceNormalizer 	NFKC normalization, prepend ▁ for whitespace 	Llama, T5
         */

        /*
PreTokenizer 	What It Does 	Used By
RegexPreTokenizer 	Split by regex pattern 	Tiktoken (GPT-4o, GPT-4)
RobertaPreTokenizer 	GPT-2/RoBERTa byte-level splitting 	GPT-2, RoBERTa
WhiteSpace 	Split on whitespace 	Simple models
CompositePreTokenizer 	Chain multiple pre-tokenizers 	Custom pipelines
        */

        /*
         Algorithm 	How It Tokenizes 	.NET Class
Tiktoken (optimized BPE) 	Pre-computed merges, very fast 	TiktokenTokenizer
BPE 	Iteratively merge most frequent pairs 	BpeTokenizer
WordPiece 	Greedy longest-subword-first 	WordPieceTokenizer, BertTokenizer
SentencePiece (Unigram) 	Probabilistic: most likely segmentation 	SentencePieceTokenizer, LlamaTokenizer
        */

        /*
Model 	Tokenizer Class 	Factory Method 	Data Package
GPT-4o, o1, o3-mini 	TiktokenTokenizer 	CreateForModel("gpt-4o") 	O200kBase
GPT-4, GPT-3.5 	TiktokenTokenizer 	CreateForModel("gpt-4") 	Cl100kBase
GPT-2 	TiktokenTokenizer 	CreateForModel("gpt-2") 	Gpt2
Llama 2, Llama 3 	LlamaTokenizer 	Create(modelStream) 	Download .model from HF
T5, mBART 	SentencePieceTokenizer 	Create(modelStream) 	Download .model from HF
BERT, DistilBERT 	BertTokenizer 	Create(vocabStream) 	Download vocab.txt from HF
CodeGen 	CodeGenTokenizer 	Constructor 	Vocab + merges files
Phi-2 	Phi2Tokenizer 	Constructor 	Vocab + merges files
         */

        /*
         BPE

// BpeTokenizer — generic BPE
BpeTokenizer tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);

// TiktokenTokenizer — OpenAI's optimized BPE variant
TiktokenTokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
         */

        /*
        WordPiece

WordPieceTokenizer tokenizer = WordPieceTokenizer.Create("vocab.txt");
// If "xyzzy" has no matching subwords → becomes [UNK]
         */


        /*
         // WordPieceTokenizer — raw WordPiece
WordPieceTokenizer tokenizer = WordPieceTokenizer.Create("vocab.txt",
    new WordPieceOptions
    {
        UnknownToken = "[UNK]",
        ContinuingSubwordPrefix = "##",
        MaxInputCharsPerWord = 200
    });

// BertTokenizer — WordPiece with BERT-specific pipeline
BertTokenizer tokenizer = BertTokenizer.Create(vocabStream,
    new BertOptions
    {
        LowerCaseBeforeTokenization = true,
        SeparatorToken = "[SEP]",
        ClassificationToken = "[CLS]",
        PaddingToken = "[PAD]",
        UnknownToken = "[UNK]"
    });
         */

        /*
         // SentencePiece with Unigram model (most common configuration for T5, etc.)
SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(modelStream);

// Llama uses SentencePiece with BPE (not Unigram)
LlamaTokenizer tokenizer = LlamaTokenizer.Create(modelStream);
        */

        /*
         // Both use SentencePiece .model files, but may use different internal algorithms
SentencePieceTokenizer t5Tokenizer = SentencePieceTokenizer.Create(t5ModelStream);     // likely Unigram
LlamaTokenizer llamaTokenizer = LlamaTokenizer.Create(llamaModelStream);              // BPE inside
        */
    }
}
