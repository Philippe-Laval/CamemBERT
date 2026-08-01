using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT
{
    public static class TiktokenTokenizerFactory
    {
        public static TiktokenTokenizer Create()
        {
            TiktokenTokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
            return tokenizer;
        }
    }
}
