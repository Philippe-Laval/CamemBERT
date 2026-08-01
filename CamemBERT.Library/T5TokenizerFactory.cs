using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/04-sentencepiece-llama-tokenizer.md
// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/cookbook/sentencepiece-tokenizer.cs

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
    public static class T5TokenizerFactory
    {
        public static SentencePieceTokenizer Create()
        {

            using var modelStream = GetT5TokenizerModelStream();

            if (modelStream == null)
                throw new InvalidOperationException("tokenizer.model stream is null");

            SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(
                modelStream,
                addBeginningOfSentence: false,  // T5 typically doesn't use BOS
                addEndOfSentence: true);        // T5 adds </s> at end

            return tokenizer;
        }


        // Charge le contenu embarqué de T5/spiece.model comme flux.
        public static Stream? GetT5TokenizerModelStream()
        {
            // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
            // Dans ce projet, le root namespace est "CamemBERT" et le fichier est T5\spiece.model
            string resourceName = "CamemBERT.T5.spiece.model";
            Assembly asm = Assembly.GetExecutingAssembly();
            Stream? stream = asm.GetManifestResourceStream(resourceName);
            return stream;
        }
    }
}
