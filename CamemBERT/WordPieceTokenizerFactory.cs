using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;



// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/05-wordpiece-bert-tokenizer.md
// Download from HuggingFace: https://huggingface.co/google-bert/bert-base-uncased/resolve/main/vocab.txt

// Critical difference from BPE: If WordPiece can't decompose a word into known subwords, the entire word becomes [UNK]

namespace CamemBERT
{
    public class WordPieceTokenizerFactory
    {
        public static WordPieceTokenizer Create()
        {
            using var modelStream = GetBertBaseUncasedVocabStream();

            if (modelStream == null)
                throw new InvalidOperationException("vocab.txt stream is null");

            WordPieceTokenizer tokenizer = WordPieceTokenizer.Create(modelStream,
                new WordPieceOptions
                {
                    UnknownToken = "[UNK]",
                    ContinuingSubwordPrefix = "##",
                    MaxInputCharsPerWord = 200
                });

            return tokenizer;
        }


        // Charge le contenu embarqué de BertBaseUncased\vocab.txt comme flux.
        public static Stream? GetBertBaseUncasedVocabStream()
        {
            // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
            // Dans ce projet, le root namespace est "CamemBERT" et le fichier est BertBaseUncased\vocab.txt
            string resourceName = "CamemBERT.BertBaseUncased.vocab.txt";
            Assembly asm = Assembly.GetExecutingAssembly();
            Stream? stream = asm.GetManifestResourceStream(resourceName);
            return stream;
        }
    }
}
