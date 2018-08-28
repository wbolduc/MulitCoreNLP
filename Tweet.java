/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package multicorenlp;

import edu.stanford.nlp.pipeline.CoreDocument;
import java.util.List;

/**
 *
 * @author wbolduc
 */
public class Tweet {
        
    public final String author;
    public final CoreDocument text;
    public final long tweetID;
    private List<SVO> svos;

    public Tweet(long tweetID, String author, CoreDocument text) {
        this.author = author;
        this.text = text;
        this.tweetID = tweetID;
    }

    public Tweet(long tweetID, String author, CoreDocument text, List<SVO> svos) {
        this.author = author;
        this.text = text;
        this.tweetID = tweetID;
        this.svos = svos;
    }

    public List<SVO> getSvos() {
        return svos;
    }

    public void setSvos(List<SVO> svos) {
        this.svos = svos;
    }
    
    public String tweetToCSV()
    {
        StringBuilder sb = new StringBuilder();
        
        svos.forEach(svo ->{
            sb.append(author);
            sb.append(",");
            sb.append(tweetID);
            sb.append(",");
            sb.append(svo.getSubject());
            sb.append(",");
            sb.append(svo.getVerb());
            sb.append(",");
            sb.append(svo.getObject());
            sb.append(",");
            sb.append(svo.isNegated());
            sb.append("\n");
        });
        return sb.toString();
    }
}
