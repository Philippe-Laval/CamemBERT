using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;

namespace CamemBERT
{
    public class BertTokenizerFactory
    {
        public static BertTokenizer Create()
        {
            using var modelStream = GetBertBaseUncasedVocabStream();

            if (modelStream == null)
                throw new InvalidOperationException("vocab.txt stream is null");

            BertTokenizer tokenizer = BertTokenizer.Create(modelStream,
                new BertOptions
                {
                    LowerCaseBeforeTokenization = true,
                    ClassificationToken = "[CLS]",
                    SeparatorToken = "[SEP]",
                    PaddingToken = "[PAD]"
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
