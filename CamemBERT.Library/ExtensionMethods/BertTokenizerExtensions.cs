using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/examples/05-bert-embedding-preprocessing.cs

namespace CamemBERT.Library.ExtensionMethods;

public static class BertTokenizerExtensions
{
    public static (int[][] InputIds, int[][] AttentionMasks) Preprocessing(this BertTokenizer tokenizer, string[] sentences, int? caps = null)
    {
        // BERT embedding preprocessing pipeline:
        // BertTokenizer tokenizer = BertTokenizerFactory.Create(
        //    new BertOptions { LowerCaseBeforeTokenization = true });

        // 1. Tokenize all sentences
        var encoded = sentences
            .Select(s => tokenizer.EncodeToIds(s))
            .ToList();

        // 2. Find max length
        int padTo = encoded.Max(e => e.Count);

        if (caps.HasValue)
        {
            padTo = Math.Min(padTo, caps.Value);
        }

        // 3. Pad and create attention masks
        int sentencesLength = sentences.Length;


        var inputIds = new int[sentencesLength][];
        var attentionMasks = new int[sentencesLength][];

        for (int i = 0; i < sentencesLength; i++)
        {
            // Initialize arrays for input IDs and attention masks with [PAD] = 0 in BERT
            inputIds[i] = new int[padTo];
            attentionMasks[i] = new int[padTo];

            var ids = encoded[i];
            int copyLen = Math.Min(ids.Count, padTo);

            for (int j = 0; j < copyLen; j++)
            {
                inputIds[i][j] = ids[j];
                attentionMasks[i][j] = 1; // 1 : Real token - 0 : Padding
            }
            
        }

        return (inputIds, attentionMasks);
    }
}
