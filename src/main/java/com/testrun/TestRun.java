package com.testrun;

import opennlp.tools.sentdetect.*;
import opennlp.tools.tokenize.*;
import opennlp.tools.postag.*;
import opennlp.tools.lemmatizer.*;

import java.io.*;
import java.util.Arrays;

public class TestRun {

    private SentenceDetectorME sentenceDetector;
    private TokenizerME tokenizer;
    private POSTaggerME posTagger;
    private LemmatizerME lemmatizer;

    public TestRun() throws IOException {
        System.out.println("Loading OpenNLP model...");

        // 1. Sentence Detector Model
        try (InputStream modelIn = getClass().getResourceAsStream("/models/opennlp-en-ud-ewt-sentence-1.2-2.5.0.bin")) {
            if (modelIn == null) throw new IOException("Sentence Detector model not found.");
            SentenceModel model = new SentenceModel(modelIn);
            sentenceDetector = new SentenceDetectorME(model);
        }

        // 2. Tokenizer Model
        try (InputStream modelIn = getClass().getResourceAsStream("/models/opennlp-en-ud-ewt-tokens-1.2-2.5.0.bin")) {
            if (modelIn == null) throw new IOException("Tokenizer model not found.");
            TokenizerModel model = new TokenizerModel(modelIn);
            tokenizer = new TokenizerME(model);
        }

        // 3. POS Tagger Model
        try (InputStream modelIn = getClass().getResourceAsStream("/models/opennlp-en-ud-ewt-pos-1.2-2.5.0.bin")) {
            if (modelIn == null) throw new IOException("POS Tagger model not found.");
            POSModel model = new POSModel(modelIn);
            posTagger = new POSTaggerME(model);
        }

        // 4. Lemmatizer Model
        try (InputStream modelIn = getClass().getResourceAsStream("/models/opennlp-en-ud-ewt-lemmas-1.2-2.5.0.bin")) {
            if (modelIn == null) throw new IOException("Lemmatizer model not found.");
            LemmatizerModel model = new LemmatizerModel(modelIn);
            lemmatizer = new LemmatizerME(model);
        }

        System.out.println("OpenNLP is loaded.");
    }

    public void processText(String text) {
        System.out.println("\n--- text processing: " + text + " ---");

        // 1. sentence chopping
        String[] sentences = sentenceDetector.sentDetect(text);
        System.out.println("Sentences: :");
        Arrays.stream(sentences).forEach(s -> System.out.println("  - " + s));

        for (String sentence : sentences) {
            // 2. tokenization
            String[] tokens = tokenizer.tokenize(sentence);
            System.out.println("\n  Tokenization results (" + sentence + "):");
            Arrays.stream(tokens).forEach(t -> System.out.print(t + " | "));
            System.out.println();

            // 3. POS
            String[] tags = posTagger.tag(tokens);
            System.out.println("  POS results:");
            for (int i = 0; i < tokens.length; i++) {
                System.out.print(tokens[i] + "/" + tags[i] + " ");
            }
            System.out.println();

            // 4. Lemma
            String[] lemmas = lemmatizer.lemmatize(tokens, tags);
            System.out.println("  Lemmas results:");
            for (int i = 0; i < tokens.length; i++) {
                System.out.print(tokens[i] + " -> " + lemmas[i] + " ");
            }
            System.out.println();
        }
    }

    public static void main(String[] args) {
        try {
            TestRun nlpProcessor = new TestRun();

            String text1 = "Hi. How are you today? I hope you are doing well. This is a test sentence.";
            String text2 = "OpenNLP is an Apache project for natural language processing. It provides many tools.";

            nlpProcessor.processText(text1);
            nlpProcessor.processText(text2);

        } catch (IOException e) {
            System.err.println("Error loading OpenNLP model: " + e);
            e.printStackTrace();
        }
    }
}