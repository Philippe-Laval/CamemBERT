using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/04-sentencepiece-llama-tokenizer.md
//
// https://huggingface.co/meta-llama/Llama-2-7b/resolve/main/tokenizer.model

//Why

// Tokenize text for Llama 2, Llama 3 and other Meta models
// Support T5, XLNet, ALBERT, mBART models
// Handle multilingual text — SentencePiece treats input as raw bytes, supporting any language
// Enable reversible tokenization — the ▁ space marker means you can perfectly reconstruct the original text


/*
SentencePieceTokenizer and LlamaTokenizer handle models trained with Google's SentencePiece framework — including Meta's Llama family, Google's T5, XLNet, ALBERT, and mBART.

Key insight: SentencePiece is a framework, not an algorithm. It can run either BPE or Unigram underneath. Llama uses SentencePiece with BPE. T5 uses SentencePiece with Unigram. 
 */

namespace CamemBERT;

public static class LlamaTokenizerFactory
{
    public static LlamaTokenizer Create()
    {
        using var modelStream = GetLlamaTokenizerModelStream();

        if (modelStream == null)
            throw new InvalidOperationException("tokenizer.model stream is null");

        LlamaTokenizer tokenizer = LlamaTokenizer.Create(
            modelStream,
            addBeginOfSentence: true,   // Add <s> at start
            addEndOfSentence: false);   // Don't add </s> at end

        return tokenizer;
    }

    // Charge le contenu embarqué de Llama/llama-tokenizer.model comme flux.
    public static Stream? GetLlamaTokenizerModelStream()
    {
        // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
        // Dans ce projet, le root namespace est "CamemBERT" et le fichier est Llama\tokenizer.model
        string resourceName = "CamemBERT.Llama.tokenizer.model";
        Assembly asm = Assembly.GetExecutingAssembly();
        Stream? stream = asm.GetManifestResourceStream(resourceName);
        return stream;
    }
}
