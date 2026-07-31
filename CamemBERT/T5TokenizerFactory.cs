using System;
using System.Collections.Generic;
using System.Text;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/04-sentencepiece-llama-tokenizer.md

// SentencePiece uses protobuf .model files, not vocab.json/merges.txt
// https://huggingface.co/google-t5/t5-base/resolve/main/spiece.model


/*
Feature 	        SentencePieceTokenizer 	                LlamaTokenizer
Base class 	        Tokenizer 	                            SentencePieceTokenizer
Algorithm inside 	BPE or Unigram (depends on .model) 	    BPE (Llama-specific)
Use for 	        T5, XLNet, mBART, ALBERT 	            Llama 2, Llama 3
Special handling 	Generic SentencePiece 	                Llama-specific byte fallback 
 */

// A T5 .model uses Unigram

namespace CamemBERT
{
    internal class T5TokenizerFactory
    {
    }
}
