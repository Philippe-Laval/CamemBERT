using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

// For a real model, download from HuggingFace:
//   vocab.json: https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json
//   merges.txt: https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt

namespace CamemBERT
{
    public static class BpeTokenizerFactory
    {

        public static BpeTokenizer Create()
        {
            using var vocabStream = GetVocabJsonStream();
            using var mergesStream = GetMergesTxtStream();

            if (vocabStream == null)
                throw new InvalidOperationException("vocab.json stream is null");
            if (mergesStream == null)
                throw new InvalidOperationException("merges.txt stream is null");

            BpeTokenizer bpeTokenizer = BpeTokenizer.Create(vocabStream, mergesStream);
            return bpeTokenizer;
        }

        public static BpeTokenizer Create(Normalizer? normalizer = null)
        {
            using var vocabStream = GetVocabJsonStream();
            using var mergesStream = GetMergesTxtStream();

            if (vocabStream == null)
                throw new InvalidOperationException("vocab.json stream is null");
            if (mergesStream == null)
                throw new InvalidOperationException("merges.txt stream is null");

            BpeTokenizer bpeTokenizer = BpeTokenizer.Create(vocabStream, mergesStream, normalizer: normalizer);
            return bpeTokenizer;
        }



        public static BpeTokenizer CreateWithCustomOptions()
        {
            using var vocabStream = GetVocabJsonStream();
            using var mergesStream = GetMergesTxtStream();

            if (vocabStream == null)
                throw new InvalidOperationException("vocab.json stream is null");
            if (mergesStream == null)
                throw new InvalidOperationException("merges.txt stream is null");

            BpeTokenizer tokenizer = BpeTokenizer.Create(
                vocabStream,
                mergesStream,
                unknownToken: "<unk>",
                specialTokens: new Dictionary<string, int>
                {
                    ["<s>"] = 0,      // Begin of sequence
                    ["</s>"] = 1,     // End of sequence
                    ["<unk>"] = 2,    // Unknown
                    ["<pad>"] = 3,    // Padding
                });
            return tokenizer;
        }

        // Charge le contenu embarqué de gpt2/vocab.json comme flux.
        public static Stream? GetVocabJsonStream()
        {
            // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
            // Dans ce projet, le root namespace est "CamemBERT" et le fichier est gpt2\vocab.json
            string resourceName = "CamemBERT.gpt2.vocab.json";
            Assembly asm = Assembly.GetExecutingAssembly();
            Stream? stream = asm.GetManifestResourceStream(resourceName);
            return stream;
        }

        // Charge le contenu embarqué de gpt2/merges.txt comme flux.
        public static Stream? GetMergesTxtStream()
        {
            // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
            // Dans ce projet, le root namespace est "CamemBERT" et le fichier est gpt2\merges.txt
            string resourceName = "CamemBERT.gpt2.merges.txt";
            Assembly asm = Assembly.GetExecutingAssembly();
            Stream? stream = asm.GetManifestResourceStream(resourceName);
            return stream;
        }
    }
}
