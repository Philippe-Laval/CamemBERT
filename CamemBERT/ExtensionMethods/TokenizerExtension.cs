using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

// MIT Licence - Luis Quintanilla
// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/examples/04-text-chunking-pipeline.cs

namespace CamemBERT.ExtensionMethods
{
    public static class TokenizerExtension
    {
        public static List<string> ChunkWithOverlap(this Tokenizer tokenizer, string text, int chunkSize, int overlapTokens)
        {
            var chunks = new List<string>();
            int position = 0;

            while (position < text.Length)
            {
                string remaining = text[position..];
                int remainingTokens = tokenizer.CountTokens(remaining);

                if (remainingTokens <= chunkSize)
                {
                    chunks.Add(remaining);
                    break;
                }

                int charIndex = tokenizer.GetIndexByTokenCount(
                    remaining, chunkSize, out _, out _);

                chunks.Add(remaining[..charIndex]);

                // Advance by (chunkSize - overlap) tokens
                int advance = chunkSize - overlapTokens;
                int advanceChars = tokenizer.GetIndexByTokenCount(
                    remaining, advance, out _, out _);
                position += advanceChars;
            }

            return chunks;
        }
    }
}
