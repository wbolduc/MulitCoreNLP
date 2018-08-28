/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multicorenlp;

import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.pipeline.CoreDocument;
import edu.stanford.nlp.pipeline.CoreSentence;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.trees.UniversalEnglishGrammaticalRelations;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.Reader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;
import maltparsetest.ConcurrentCounter;
import maltparsetest.MaltMultiParse;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;

/**
 *
 * @author wbolduc
 */
public class MultiCoreNLP implements Runnable {
    
    private final int reportInterval = 100;
    private final StanfordCoreNLP pipeline;
    private final List<Tweet> tweets;
    
    private AtomicInteger tweetsParsed;
    
    MultiCoreNLP(StanfordCoreNLP pipeline, List<Tweet> tweets, AtomicInteger counter)
    {
        this.pipeline = pipeline;
        this.tweets = new ArrayList<>(tweets);
        this.tweetsParsed = counter;
    }
    
    @Override
    public void run() {
        long threadStartTime = System.nanoTime();
        int i = 0;
        for(Tweet tweet : tweets)
        {
            //annotate
            pipeline.annotate(tweet.text);
            
            //get svos in this tweet
            List<SVO> sentSVOs = new ArrayList<>();
            for(CoreSentence sent : tweet.text.sentences())
                sentSVOs.addAll(extractSVOs(sent.dependencyParse()));
            
            //save SVOs to tweet
            tweet.setSvos(sentSVOs);
            
            if((++i) == reportInterval)
            {
                tweetsParsed.addAndGet(reportInterval);
                i = 0;
            }
        }
        System.out.println(getMessegeWithElapsed("Thread " + Long.toString(Thread.currentThread().getId()) + " finished: ", threadStartTime));
    }
    
    public static List<SVO> extractSVOs(SemanticGraph depRels)
    {
        List<SVO> svos = new ArrayList<>();
        
        Collection<TypedDependency> typedDeps = depRels.typedDependencies();
        typedDeps.forEach(typedDep ->{    
            if(typedDep.reln().getShortName().startsWith("nsub"))   //this dep is a subject
            {
                IndexedWord verb = typedDep.gov();
                
                Boolean negated = false;//depRels.isNegatedVertex(verb);    //doesnt seem to work
                for(GrammaticalRelation gr : depRels.childRelns(verb))
                    if (gr.getShortName().equals("neg"))
                        if(negated == false)
                            negated = true;
                        else
                            negated = false;
                
                for (TypedDependency obj : typedDeps)
                {
                    String relName = obj.reln().getShortName();
                    if (obj.gov().equals(verb) && (relName.equals("dobj")/* || relName.equals("nmod")*/)) //object sharing the same verb
                    {
                        svos.add(new SVO(typedDep.dep().lemma(), verb.lemma(), obj.dep().lemma(), negated));
                    }
                }
            }
        });
        return svos;
    }
    
    
    public static void main(String[] args){
        int threadCount = 8;
        int chunkSize = 20000;
        
        String outputFile = "E:\\YorkWork\\conllStages\\SVOdata\\Twitter-MeToo-en-2017-10-17-to-31-Basic.csv";
        
        // set up pipeline properties
        Properties props = new Properties();
        // set the list of annotators to run
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner, parse, depparse, coref"); 
        // set a property for an annotator, in this case the coref annotator is being set to use the neural algorithm
        props.setProperty("coref.algorithm", "neural");
        //props.setProperty("parse.model", "englishPCFG.caseless.ser.gz");
        props.setProperty("depparse.language", "english");
        
        // build pipeline
        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
        
        // load tweets
        List<Tweet> tweets = null;
        try {
            tweets = loadAllCSVTweets("E:\\YorkWork\\conllStages\\raw\\Twitter-MeToo-en-2017-10-17-to-31.csv");//"E:\\YorkWork\\conllStages\\raw\\NASAClimate-all-tweets.csv"
        } catch (IOException ex) {
            System.out.println("Could not load file.");
            return;
        }
        
        // split tweets
        int tweetCount = tweets.size();
        int chunk = tweetCount / threadCount;
        
        
        
        
        System.out.println("Loaded " + Integer.toString(tweetCount) + " tweets");
        
        // create progress counter
        AtomicInteger tweetsParsed = new AtomicInteger(0);
        Thread counter = new Thread(new ConcurrentCounter("Tweets parsed : ", tweetsParsed, 1000));
        
        //create runnables
        MultiCoreNLP[] runnables = new MultiCoreNLP[threadCount];
        int i;
        for(i = 0; i < threadCount - 1; i++)
        {
            runnables[i] = new MultiCoreNLP(pipeline, tweets.subList(chunk*i, chunk*(i+1)), tweetsParsed);
            System.out.println("Thread " + Integer.toString(i) + ": processing tweets " + Integer.toString(chunk*i) + " to " + Integer.toString(chunk*(i+1)));
        }
        runnables[i] = new MultiCoreNLP(pipeline, tweets.subList(chunk*i, tweetCount), tweetsParsed);
        System.out.println("Thread " + Integer.toString(i) + ": processing tweets " + Integer.toString(chunk*i) + " to " + Integer.toString(tweetCount));
            
        // create threads
        Thread[] threads = new Thread[threadCount];
        for(i = 0; i < threadCount; i++)
            threads[i] = new Thread(runnables[i]);
        
        // start threads
        System.out.println("Starting " + Integer.toString(threadCount) + " threads");
        long startTime = System.nanoTime();
        for(Thread thread : threads)
            thread.start();
   
        // Start counter
        counter.start();
        
        // join threads
        for(Thread thread : threads)
            try{thread.join();} catch (InterruptedException ignore) {}
        System.out.println(getMessegeWithElapsed("All threads finished. ", startTime));
        
        //stop counter
        counter.interrupt();
        try {
            counter.join();
        } catch (InterruptedException ignore){}
                
        //print to file
        try {
            PrintWriter writer = new PrintWriter(outputFile);
            writer.print("user_screen_name,id,subject,verb,object,is_negated\n");
            tweets.forEach(tweet -> writer.print(tweet.tweetToCSV()));
            writer.close();
        } catch (IOException ex) {
            Logger.getLogger(MaltMultiParse.class.getName()).log(Level.SEVERE, null, ex);
        }
    }
    
    public static ArrayList<Tweet> loadAllCSVTweets(String fileName) throws FileNotFoundException, IOException
    {
        ArrayList<Tweet> tweets = new ArrayList<>();
        
        Reader csvData = new FileReader(fileName);
        Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(csvData);
        
        for(CSVRecord record : records)
            tweets.add(new Tweet(Long.parseLong(record.get("id")), record.get("user_screen_name"), new CoreDocument(record.get("text"))));
        
        return tweets;
    }
    
    public static String getMessegeWithElapsed(String message, long startTime) {
        final StringBuilder sb = new StringBuilder();
        long elapsed = (System.nanoTime() - startTime) / 1000000;
        sb.append(message);
        sb.append(" : ");
        sb.append(elapsed);
        sb.append(" ms");
        return sb.toString();
    }
}
