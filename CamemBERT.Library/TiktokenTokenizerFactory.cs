using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

namespace CamemBERT
{
    public static class TiktokenTokenizerFactory
    {
        public static TiktokenTokenizer CreateGpt5()
        {
            TiktokenTokenizer gpt5 = TiktokenTokenizer.CreateForModel("gpt-5");
            return gpt5;
        }

        public static TiktokenTokenizer CreateGpt4o()
        {
            // <PackageReference Include = "Microsoft.ML.Tokenizers.Data.O200kBase" Version="2.0.0" />
            TiktokenTokenizer gpt4o = TiktokenTokenizer.CreateForModel("gpt-4o");
            return gpt4o;
        }


        public static TiktokenTokenizer CreateGpt4()
        {
            // <PackageReference Include = "Microsoft.ML.Tokenizers.Data.Cl100kBase" Version="2.0.0" />
            TiktokenTokenizer gpt4 = TiktokenTokenizer.CreateForModel("gpt-4");
            return gpt4;
        }

    }
}
