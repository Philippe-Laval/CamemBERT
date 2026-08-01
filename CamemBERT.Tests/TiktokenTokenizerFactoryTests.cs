using Microsoft.ML.Tokenizers;
using Newtonsoft.Json.Linq;
using System;
using System.Collections.Generic;
using System.Text;
using static System.Net.Mime.MediaTypeNames;

// https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/docs/06-encoding-decoding.md

namespace CamemBERT.Library.Tests;

[TestClass]
public sealed class TiktokenTokenizerFactoryTests
{
    [TestMethod]
    public void TestEncodeToIds_Basic()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "The quick brown fox jumps over the lazy dog.";

        // String overload
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);
        Console.WriteLine($"Token count: {ids.Count}");              // Token count: 10
        Console.WriteLine($"IDs: [{string.Join(", ", ids)}]");

        // Span overload (avoids string allocation if you already have a span)
        ReadOnlySpan<char> span = text.AsSpan();
        IReadOnlyList<int> ids2 = tokenizer.EncodeToIds(span);
    }

    [TestMethod]
    public void TestEncodeToIds_WithMaxTokenLimit()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "The quick brown fox jumps over the lazy dog.";

        // Encode at most 5 tokens, find out how many characters were consumed
        IReadOnlyList<int> truncatedIds = tokenizer.EncodeToIds(
        text,
        maxTokenCount: 5,
        out string? normalizedText,
        out int charsConsumed);

        Console.WriteLine($"Truncated to {truncatedIds.Count} tokens");   // 5
        Console.WriteLine($"Characters consumed: {charsConsumed}");        // ~23
        Console.WriteLine($"Text consumed: '{text[..charsConsumed]}'");
    }

    [TestMethod]
    public void TestEncodeToToken_Detailed_Token_Information()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "The quick brown fox jumps over the lazy dog.";

        IReadOnlyList<EncodedToken> tokens = tokenizer.EncodeToTokens(text, out string? normalized);

        foreach (EncodedToken token in tokens)
        {
            string sourceText = text[token.Offset];  // Extract the original text this token came from
            Console.WriteLine($"  Token: '{token.Value}' | ID: {token.Id} | " +
                          $"Offset: [{token.Offset.Start}..{token.Offset.End}] | " +
                          $"Source: '{sourceText}'");
        }
    }


    [TestMethod]
    public void TestCountTokens()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "The quick brown fox jumps over the lazy dog.";

        // When you only need the count, CountTokens is more efficient
        // (doesn't allocate the token ID list)
        int count = tokenizer.CountTokens(text);
        Console.WriteLine($"Token count: {count}");  // Token count: 10

        // Equivalent but less efficient:
        int countAlt = tokenizer.EncodeToIds(text).Count;  // Allocates the list
    }

    [TestMethod]
    public void TestDecode()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "Hello, world!";

        // Encode
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(text);

        // Decode
        string? decoded = tokenizer.Decode(ids);
        Console.WriteLine($"Decoded: '{decoded}'");  // Decoded: 'Hello, world!'

        // Decode arbitrary IDs (e.g., from model output)
        string? fromModel = tokenizer.Decode(new[] { 15339, 11, 1917, 0 });
        Console.WriteLine($"From model: '{fromModel}'");  // From model: ' awesome, down!'
    }


    [TestMethod]
    public void TestTokenOffsets()
    {
        /*
EncodedToken.Offset is a Range that maps each token back to its position in the original (or normalized) text. This is invaluable for:

Highlighting which part of the text each token came from
Named Entity Recognition — mapping token-level labels back to text spans
Debugging — understanding token boundaries  
         */

        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "Microsoft's AI tokenizer";
        var tokens = tokenizer.EncodeToTokens(text, out _);

        foreach (var token in tokens)
        {
            int start = token.Offset.Start.Value;
            int end = token.Offset.End.Value;
            string highlight = text[start..end];
            Console.WriteLine($"  [{start,2}..{end,2}] '{token.Value}' → '{highlight}'");
        }

    }

    [TestMethod]
    public void TestBypassingPipelineStages()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();
        string text = "The quick brown fox jumps over the lazy dog.";

        // Skip normalization (text already preprocessed)
        var ids = tokenizer.EncodeToIds(text,
            considerNormalization: false);

        // Skip both normalization and pre-tokenization
        var ids2 = tokenizer.EncodeToIds(text,
            considerNormalization: false,
            considerPreTokenization: false);
    }


    [TestMethod]
    public void TestVisualTokenizerComparison()
    {
        // https://github.com/luisquintanilla/dotnet-tokenizers-guide/blob/main/examples/07-tokenizer-comparison-visual.cs

        Console.WriteLine("═══ Visual Tokenizer Comparison ═══\n");

        TiktokenTokenizer gpt4o = TiktokenTokenizerFactory.CreateGpt4o();
        TiktokenTokenizer gpt4 = TiktokenTokenizerFactory.CreateGpt4();

        // ── 1. Side-by-side token boundaries ────────────────────────────────────────
        Console.WriteLine("═══ 1. Token Boundaries — Same Text, Different Tokenizers ═══\n");

        string[] texts = [
            "Hello, world!",
            "Tokenization is the first step in NLP.",
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1)",
            "The café served espresso and crème brûlée.",
            "🤖 AI is transforming 🌍 the world!",
            "Microsoft.ML.Tokenizers.TiktokenTokenizer",
        ];

        foreach (string text in texts)
        {
            Console.WriteLine($"Text: \"{text}\"\n");

            // GPT-4o
            var _gpt4oTokens = gpt4o.EncodeToTokens(text, out _);
            Console.Write("  GPT-4o:  |");
            foreach (var t in _gpt4oTokens)
            {
                string val = t.Value.Replace("\n", "\\n");
                Console.Write($"{val}|");
            }
            Console.WriteLine($"  ({_gpt4oTokens.Count} tokens)");

            // GPT-4
            var _gpt4Tokens = gpt4.EncodeToTokens(text, out _);
            Console.Write("  GPT-4:   |");
            foreach (var t in _gpt4Tokens)
            {
                string val = t.Value.Replace("\n", "\\n");
                Console.Write($"{val}|");
            }
            Console.WriteLine($"  ({_gpt4Tokens.Count} tokens)");

            Console.WriteLine();
        }

        // ── 2. Token ID comparison ──────────────────────────────────────────────────
        Console.WriteLine("═══ 2. Token IDs — Same Word, Different Vocabularies ═══\n");

        string[] words = ["Hello", " the", "AI", " world", "function", "\n"];

        Console.WriteLine($"  {"Token",-12} {"GPT-4o ID",10} {"GPT-4 ID",10}  Same?");
        Console.WriteLine($"  {new string('-', 12)} {new string('-', 10)} {new string('-', 10)}  {new string('-', 5)}");

        foreach (string word in words)
        {
            var o = gpt4o.EncodeToIds(word);
            var c = gpt4.EncodeToIds(word);

            string display = word.Replace("\n", "\\n").Replace(" ", "·");
            string oId = o.Count == 1 ? o[0].ToString() : $"[{string.Join(",", o)}]";
            string cId = c.Count == 1 ? c[0].ToString() : $"[{string.Join(",", c)}]";
            bool same = o.SequenceEqual(c);

            Console.WriteLine($"  {$"\"{display}\"",-12} {oId,10} {cId,10}  {(same ? "✅" : "❌")}");
        }

        Console.WriteLine("\n  Note: Same token → different IDs in different vocabularies.");
        Console.WriteLine("  Never mix token IDs from different tokenizers!\n");

        // ── 3. Efficiency comparison ────────────────────────────────────────────────
        Console.WriteLine("═══ 3. Token Efficiency by Content Type ═══\n");

        var contentTypes = new (string Type, string Sample)[]
        {
("English prose",   "The quick brown fox jumps over the lazy dog on a sunny afternoon."),
("Python code",     "def quicksort(arr): return [] if not arr else quicksort([x for x in arr[1:] if x <= arr[0]]) + [arr[0]] + quicksort([x for x in arr[1:] if x > arr[0]])"),
("JSON",            """{"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]}"""),
("SQL",             "SELECT u.name, COUNT(o.id) AS order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.name HAVING COUNT(o.id) > 5 ORDER BY order_count DESC"),
("Chinese",         "人工智能正在以前所未有的速度改变着我们的世界和日常生活。"),
("Math notation",   "∀x ∈ ℝ: |sin(x)| ≤ 1 ∧ ∃n ∈ ℤ: sin(nπ) = 0"),
("Mixed emoji",     "🎉 Party 🎊 time! 🎈 Let's 🥳 celebrate 🎁 together! 🎄"),
("URL",             "https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.tokenizers"),
        };

        Console.WriteLine($"  {"Content Type",-18} {"Chars",6} {"GPT-4o",7} {"GPT-4",7} {"4o tok/ch",10} {"4 tok/ch",10}");
        Console.WriteLine($"  {new string('-', 18)} {new string('-', 6)} {new string('-', 7)} {new string('-', 7)} {new string('-', 10)} {new string('-', 10)}");

        foreach (var (type, sample) in contentTypes)
        {
            int oCount = gpt4o.CountTokens(sample);
            int cCount = gpt4.CountTokens(sample);
            double oRatio = (double)oCount / sample.Length;
            double cRatio = (double)cCount / sample.Length;

            Console.WriteLine($"  {type,-18} {sample.Length,6} {oCount,7} {cCount,7} {oRatio,10:F3} {cRatio,10:F3}");
        }

        // ── 4. Algorithm comparison reference ───────────────────────────────────────
        Console.WriteLine("\n\n═══ 4. Algorithm Comparison Reference ═══\n");

        Console.WriteLine("""
┌──────────────┬───────────────┬───────────────┬────────────────────┐
│ Property     │ Tiktoken/BPE  │ WordPiece     │ SentencePiece      │
├──────────────┼───────────────┼───────────────┼────────────────────┤
│ Used by      │ GPT-4o, GPT-4 │ BERT          │ Llama, T5          │
│ Space marker │ Leading space │ None (## pfx) │ ▁ prefix           │
│ OOV handling │ Byte fallback │ Entire → [UNK]│ Byte/char fallback │
│ Special toks │ <|endoftext|> │ [CLS] [SEP]   │ <s> </s>           │
│ Normalization│ None (GPT)    │ Lowercase     │ NFKC + ▁           │
│ .NET class   │ TiktokenTok.  │ BertTokenizer │ LlamaTokenizer     │
└──────────────┴───────────────┴───────────────┴────────────────────┘

Observations:
• GPT-4o (o200k_base) is generally more token-efficient than GPT-4 (cl100k_base)
• Non-Latin scripts are less token-efficient across all tokenizers
• Code tends to have higher token density than prose
• Emoji can consume multiple tokens due to Unicode encoding
""");

        // ── 5. Llama / BERT patterns ────────────────────────────────────────────────
        Console.WriteLine("\n═══ 5. Llama & BERT Comparison Patterns ═══\n");

       
        LlamaTokenizer llama = LlamaTokenizerFactory.Create();
        BertTokenizer bert = BertTokenizerFactory.Create();

        string text2 = "Hello, world!";

        // GPT-4o:  |Hello|,| world|!|             → 4 tokens
        // Llama:   |<s>|▁Hello|,|▁world|!|        → 5 tokens (with BOS)
        // BERT:    |[CLS]|hello|,|world|!|[SEP]|   → 6 tokens (lowercased, with CLS/SEP)


        // GPT-4
        var gpt4Tokens = gpt4.EncodeToTokens(text2, out _);
        Console.Write("  GPT-4:   |");
        foreach (var t in gpt4Tokens)
        {
            string val = t.Value.Replace("\n", "\\n");
            Console.Write($"{val}|");
        }
        Console.WriteLine($"  ({gpt4Tokens.Count} tokens)");

        Console.WriteLine();

        // GPT-4o
        var gpt4oTokens = gpt4o.EncodeToTokens(text2, out _);
        Console.Write("  GPT-4o:  |");
        foreach (var t in gpt4oTokens)
        {
            string val = t.Value.Replace("\n", "\\n");
            Console.Write($"{val}|");
        }
        Console.WriteLine($"  ({gpt4oTokens.Count} tokens)");

        Console.WriteLine();

        // Llama
        var llamaTokens = llama.EncodeToTokens(text2, out _);
        Console.Write("  Llama:   |");
        foreach (var t in llamaTokens)
        {
            string val = t.Value.Replace("\n", "\\n");
            Console.Write($"{val}|");
        }
        Console.WriteLine($"  ({llamaTokens.Count} tokens)");

        Console.WriteLine();

        // BERT
        var bertTokens = bert.EncodeToTokens(text2, out _);
        Console.Write("  BERT:    |");
        foreach (var t in bertTokens)
        {
            string val = t.Value.Replace("\n", "\\n");
            Console.Write($"{val}|");
        }
        Console.WriteLine($"  ({bertTokens.Count} tokens)");

        Console.WriteLine();


        // Key differences:
        // - GPT-4o: leading space in " world", no special tokens for plain text
        // - Llama: ▁ prefix for word starts, <s> BOS token
        // - BERT: lowercased, ## for subwords, [CLS]/[SEP] added automatically

    }


    /// <summary>
    /// GPT-5 tokenizer
    /// </summary>
    [TestMethod]
    public void TestCountTokens_gpt5()
    {
        // Initialize the tokenizer for the gpt-5 model.
        Tokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt5();

        string source = "Text tokenization is the process of splitting a string into a list of tokens.";

        // Count the tokens in the text.
        Console.WriteLine($"Tokens: {tokenizer.CountTokens(source)}");
        // Output: Tokens: 16

        // Encode text to token IDs.
        IReadOnlyList<int> ids = tokenizer.EncodeToIds(source);
        Console.WriteLine($"Token IDs: {string.Join(", ", ids)}");
        // Output: Token IDs: 1279, 6602, 2860, 382, 290, 2273, 328, 87130, 261, 1621, 1511, 261, 1562, 328, 20290, 13

        // Decode token IDs back to text.
        string? decoded = tokenizer.Decode(ids);
        Console.WriteLine($"Decoded: {decoded}");
        // Output: Decoded: Text tokenization is the process of splitting a string into a list of tokens.
    }


    [TestMethod]
    public void TestGetIndexByTokenCountFromEnd_gpt5()
    {
        Tokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt5();

        string source = "Text tokenization is the process of splitting a string into a list of tokens.";

        // Get the last 5 tokens from the text.
        var trimIndex = tokenizer.GetIndexByTokenCountFromEnd(source, 5, out string? processedText, out _);
        processedText ??= source;
        Console.WriteLine($"Last 5 tokens: {processedText.Substring(trimIndex)}");
        // Output: Last 5 tokens:  a list of tokens.

        // Get the first 5 tokens from the text.
        trimIndex = tokenizer.GetIndexByTokenCount(source, 5, out processedText, out _);
        processedText ??= source;
        Console.WriteLine($"First 5 tokens: {processedText.Substring(0, trimIndex)}");
        // Output: First 5 tokens: Text tokenization is the
    }

    [TestMethod]
    public void test1()
    {
        TiktokenTokenizer tokenizer = TiktokenTokenizerFactory.CreateGpt4o();

        IReadOnlyList<int> ids = tokenizer.EncodeToIds("Hello, world!");
        Console.WriteLine($"Token count: {ids.Count}");     // Token count: 4
        Console.WriteLine($"Token IDs: [{string.Join(", ", ids)}]");  // Token IDs: [13225, 11, 2375, 0]

        string decoded = tokenizer.Decode(ids)!;
        Console.WriteLine($"Decoded: {decoded}");            // Decoded: Hello, world!
    }




    /*
Normalizer 	What It Does 	Used By 
BertNormalizer 	Lowercase, strip accents, clean control chars, handle Chinese chars 	BERT
LowerCaseNormalizer 	Lowercases all text 	Custom
UpperCaseNormalizer 	Uppercases all text 	Custom
SentencePieceNormalizer 	NFKC normalization, prepend ▁ for whitespace 	Llama, T5
     */

    /*
PreTokenizer 	What It Does 	Used By
RegexPreTokenizer 	Split by regex pattern 	Tiktoken (GPT-4o, GPT-4)
RobertaPreTokenizer 	GPT-2/RoBERTa byte-level splitting 	GPT-2, RoBERTa
WhiteSpace 	Split on whitespace 	Simple models
CompositePreTokenizer 	Chain multiple pre-tokenizers 	Custom pipelines
    */

    /*
     Algorithm 	How It Tokenizes 	.NET Class
Tiktoken (optimized BPE) 	Pre-computed merges, very fast 	TiktokenTokenizer
BPE 	Iteratively merge most frequent pairs 	BpeTokenizer
WordPiece 	Greedy longest-subword-first 	WordPieceTokenizer, BertTokenizer
SentencePiece (Unigram) 	Probabilistic: most likely segmentation 	SentencePieceTokenizer, LlamaTokenizer
    */

    /*
Model 	Tokenizer Class 	Factory Method 	Data Package
GPT-4o, o1, o3-mini 	TiktokenTokenizer 	CreateForModel("gpt-4o") 	O200kBase
GPT-4, GPT-3.5 	TiktokenTokenizer 	CreateForModel("gpt-4") 	Cl100kBase
GPT-2 	TiktokenTokenizer 	CreateForModel("gpt-2") 	Gpt2
Llama 2, Llama 3 	LlamaTokenizer 	Create(modelStream) 	Download .model from HF
T5, mBART 	SentencePieceTokenizer 	Create(modelStream) 	Download .model from HF
BERT, DistilBERT 	BertTokenizer 	Create(vocabStream) 	Download vocab.txt from HF
CodeGen 	CodeGenTokenizer 	Constructor 	Vocab + merges files
Phi-2 	Phi2Tokenizer 	Constructor 	Vocab + merges files
     */

    /*
     BPE

// BpeTokenizer — generic BPE
BpeTokenizer tokenizer = BpeTokenizer.Create(vocabStream, mergesStream);

// TiktokenTokenizer — OpenAI's optimized BPE variant
TiktokenTokenizer tokenizer = TiktokenTokenizer.CreateForModel("gpt-4o");
     */

    /*
    WordPiece

WordPieceTokenizer tokenizer = WordPieceTokenizer.Create("vocab.txt");
// If "xyzzy" has no matching subwords → becomes [UNK]
     */


    /*
     // WordPieceTokenizer — raw WordPiece
WordPieceTokenizer tokenizer = WordPieceTokenizer.Create("vocab.txt",
new WordPieceOptions
{
    UnknownToken = "[UNK]",
    ContinuingSubwordPrefix = "##",
    MaxInputCharsPerWord = 200
});

// BertTokenizer — WordPiece with BERT-specific pipeline
BertTokenizer tokenizer = BertTokenizer.Create(vocabStream,
new BertOptions
{
    LowerCaseBeforeTokenization = true,
    SeparatorToken = "[SEP]",
    ClassificationToken = "[CLS]",
    PaddingToken = "[PAD]",
    UnknownToken = "[UNK]"
});
     */

    /*
     // SentencePiece with Unigram model (most common configuration for T5, etc.)
SentencePieceTokenizer tokenizer = SentencePieceTokenizer.Create(modelStream);

// Llama uses SentencePiece with BPE (not Unigram)
LlamaTokenizer tokenizer = LlamaTokenizer.Create(modelStream);
    */

    /*
     // Both use SentencePiece .model files, but may use different internal algorithms
SentencePieceTokenizer t5Tokenizer = SentencePieceTokenizer.Create(t5ModelStream);     // likely Unigram
LlamaTokenizer llamaTokenizer = LlamaTokenizer.Create(llamaModelStream);              // BPE inside
    */


}

