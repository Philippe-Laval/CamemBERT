using CamemBERT.Normalizers;
using Microsoft.ML.Tokenizers;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.RegularExpressions;
using System.IO;

// https://github.com/AliAlAali/Qwen3.Net.Tokenizers
// MIT License - Ali Alaali

namespace CamemBERT;

public static class Qwen3TokenizerFactory
{
    private const string QWEN3_PATTERN = @"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|[^\s\p{L}\p{N}]+|[\r\n]+|\s+(?!\S)|\s+";

    public static Tokenizer Create(string vocabsPath, string mergesPath)
    {
        var regex = new Regex(QWEN3_PATTERN);
        var specialTokens = new Dictionary<string, int>
            {
                { "<|endoftext|>", 151643 },
                { "<|im_start|>", 151644 },
                { "<|im_end|>", 151645 },
            };

        var options = new BpeOptions(vocabsPath, mergesPath)
        {
            ByteLevel = true,
            Normalizer = new NFC(),
            PreTokenizer = new RegexPreTokenizer(regex, specialTokens),
            SpecialTokens = specialTokens,
            EndOfSentenceToken = "<|endoftext|>"
        };

        return BpeTokenizer.Create(options);
    }

    private static Tokenizer CreateNotWorkingDueToInassessibleParameters()
    {
        using var vocabStream = GetVocabJsonStream();
        using var mergesStream = GetMergesTxtStream();

        if (vocabStream == null)
            throw new InvalidOperationException("vocab.json stream is null");
        if (mergesStream == null)
            throw new InvalidOperationException("merges.txt stream is null");

        var regex = new Regex(QWEN3_PATTERN);
        var specialTokens = new Dictionary<string, int>
            {
                { "<|endoftext|>", 151643 },
                { "<|im_start|>", 151644 },
                { "<|im_end|>", 151645 },
            };

        BpeTokenizer bpeTokenizer = BpeTokenizer.Create(vocabStream, 
            mergesStream,
            preTokenizer: new RegexPreTokenizer(regex, specialTokens),
            normalizer: new NFC(),
            specialTokens: specialTokens
            // ByteLevel = true
            //EndOfSentenceToken = "<|endoftext|>"
            );

        return bpeTokenizer;
    }


    public static Tokenizer Create()
    {
        using var vocabStream = GetVocabJsonStream();
        using var mergesStream = GetMergesTxtStream();

        if (vocabStream == null)
            throw new InvalidOperationException("vocab.json stream is null");
        if (mergesStream == null)
            throw new InvalidOperationException("merges.txt stream is null");

        // Ecrire vocab.json et merges.txt dans des fichiers temporaires
        string vocabsPath = Path.Combine(Path.GetTempPath(), $"qwen3_vocab_{Guid.NewGuid()}.json");
        using (var fs = File.Create(vocabsPath))
        {
            vocabStream.CopyTo(fs);
            fs.Flush();
        }

        string mergesPath = Path.Combine(Path.GetTempPath(), $"qwen3_merges_{Guid.NewGuid()}.txt");
        using (var fs2 = File.Create(mergesPath))
        {
            mergesStream.CopyTo(fs2);
            fs2.Flush();
        }

        // Utiliser la surcharge qui prend des chemins de fichiers
        Tokenizer tokenizer = Create(vocabsPath, mergesPath);

        if (File.Exists(vocabsPath))
            File.Delete(vocabsPath);

        if (File.Exists(mergesPath))
            File.Delete(mergesPath);

        return tokenizer;
    }

    // Charge le contenu embarqué de Qwen3/vocab.json comme flux.
    public static Stream? GetVocabJsonStream()
    {
        // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
        // Dans ce projet, le root namespace est "CamemBERT" et le fichier est Qwen3\vocab.json
        string resourceName = "CamemBERT.Qwen3.vocab.json";
        Assembly asm = Assembly.GetExecutingAssembly();
        Stream? stream = asm.GetManifestResourceStream(resourceName);
        return stream;
    }

    // Retourne le contenu JSON de vocab.json comme chaîne (ou null si introuvable)
    public static string? ReadVocabJsonText()
    {
        using Stream? s = GetVocabJsonStream();
        if (s == null)
            return null;
        using StreamReader r = new StreamReader(s, Encoding.UTF8);
        return r.ReadToEnd();
    }

    // Charge le contenu embarqué de Qwen3/merges.txt comme flux.
    public static Stream? GetMergesTxtStream()
    {
        // Nom de ressource embarquée: <RootNamespace>.<folder>.<file>
        // Dans ce projet, le root namespace est "CamemBERT" et le fichier est Qwen3\merges.txt
        string resourceName = "CamemBERT.Qwen3.merges.txt";
        Assembly asm = Assembly.GetExecutingAssembly();
        Stream? stream = asm.GetManifestResourceStream(resourceName);
        return stream;
    }
}