using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Text;

// https://github.com/AliAlAali/Qwen3.Net.Tokenizers
// MIT License - Ali Alaali

namespace CamemBERT.Library.Normalizers;

internal class NFC : Normalizer
{
    public override string Normalize(string original)
    {
        return original.Normalize(NormalizationForm.FormC);
    }

    public override string Normalize(ReadOnlySpan<char> original)
    {
        return original.ToString().Normalize(NormalizationForm.FormC);
    }
}
