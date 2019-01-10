package gov.nih.nlm.citationsentiment;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.logging.Logger;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.io.XMLPredicateReader;
import gov.nih.nlm.ling.io.XMLReader;
import gov.nih.nlm.ling.sem.Predicate;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.util.FileUtils;

/**
 * Class to evaluate the rule-based sentiment citation analysis program. 
 * 
 * @author Halil Kilicoglu
 *
 */
public class RuleBasedSentimentEval {
	private static Logger log = Logger.getLogger(RuleBasedSentimentEval.class.getName());	

	private RuleBasedSentiment instance; 

	private static Map<Class<? extends SemanticItem>,List<String>> annTypes;
	private static XMLReader xmlReader;

	private static Map<String,List<CitationMention>> annoTP = new HashMap<>();
	private static Map<String,List<CitationMention>> annoFP = new HashMap<>();
	private static Map<String,List<CitationMention>> annoFN = new HashMap<>();


	public RuleBasedSentimentEval(Properties props)  throws Exception {
		xmlReader = getXMLReader();
		annTypes = Utils.getAnnotationTypes();
		instance = RuleBasedSentiment.getInstance(props);
	}

	private  XMLReader getXMLReader() {
		XMLReader reader = Utils.getXMLReader();
		reader.addAnnotationReader(Predicate.class, new XMLPredicateReader());
		return reader;
	}

	/**
	 * Processes all corpus XML files and evaluates the program results.
	 * 
	 * @param dir	The corpus directory
	 * @param props
	 * @param pw	The PrintWriter object associated with the output file
	 */
	public void  processDir(String dir,Properties props, PrintWriter pw)    {
		List<CitationMention> instances = new ArrayList<>();
		try {
			List<String> files = FileUtils.listFiles(dir, false, "xml");
			Collections.sort(files);
			int fileNum = 0;
			for (String filename: files) {
				String filenameNoExt = filename.replace(".xml", "");
				filenameNoExt = filenameNoExt.substring(filenameNoExt.lastIndexOf(File.separator)+1);
				log.info("Processing " + filenameNoExt + ":" + ++fileNum);
				List<CitationMention> fileInstances = preProcessFile(filename,props);
				for (CitationMention ins: fileInstances) {
					String gold = ins.getMetaData("goldSentiment");
					String[] scoreStrs =  instance.processMention(ins).split("[\\|]");

					String predict = scoreStrs[3];
					String cumScore = scoreStrs[4];
					String scoreStr = "";
					if (scoreStrs.length > 5)
						scoreStr = scoreStrs[5];

					pw.println(predict + "|" + gold + "|" + cumScore + "|" + scoreStr + "|" + ins.toString());
					if (predict.equals(gold)) {
						List<CitationMention> tps = annoTP.get(predict);
						if (tps == null) tps = new ArrayList<CitationMention>();
						tps.add(ins);
						annoTP.put(predict, tps);
					} else {
						List<CitationMention> fps = annoFP.get(predict);
						if (fps == null) fps = new ArrayList<CitationMention>();
						fps.add(ins);
						annoFP.put(predict, fps);
						List<CitationMention> fns = annoFN.get(gold);
						if (fns == null) fns = new ArrayList<CitationMention>();
						fns.add(ins);
						annoFN.put(gold, fns);
					}
				}
				instances.addAll(fileInstances);
			}
		} catch (IOException ie) {
			log.severe("Unable to read input files from " + dir);
		}
	}

	/**
	 * Reads a file from XML and preprocesses it for gold labels and sentiment terms. 
	 * 
	 * @param filename	The file to process
	 * @param props
	 * @return The list of citation mentions from the file
	 */
	public  List<CitationMention> preProcessFile(String filename,Properties props)  {
		Document doc = null;
		doc = xmlReader.load(filename, true,CitationFactory.class, annTypes, null);
		LinkedHashSet<SemanticItem> cms = Document.getSemanticItemsByClass(doc, CitationMention.class);
		Map<String,String> gold = new HashMap<String,String>();
		for (SemanticItem s: cms) {
			if (s instanceof CitationMention == false) continue;
			CitationMention cm = (CitationMention)s;
			String sentiment = cm.getSentiment().toString();
			gold.put(cm.getId(), sentiment);
		}
		return preProcessDocument(doc,gold,props);
	}

	private  List<CitationMention> preProcessDocument(Document doc,Map<String,String> gold, Properties properties)  {
		instance.annotateTerms(doc, properties);
		List<CitationMention> citationMentions = new ArrayList<>();
		Set<SurfaceElement> seen = new HashSet<SurfaceElement>();
		LinkedHashSet<SemanticItem> mentions = Document.getSemanticItemsByClass(doc, CitationMention.class);
		for (SemanticItem m: mentions) {
			if (m instanceof CitationMention == false) continue;
			if (m.getId().startsWith("C") == false) continue;
			CitationMention cm = (CitationMention)m;
			SurfaceElement surf = cm.getSurfaceElement();
			if (seen.contains(surf)) continue;
			String goldSentiment = gold.get(cm.getId());
			if (goldSentiment.equals("NONE")) continue;
			// current sentence only
			//		  if (cm.getContext() == null)
			cm.setContext(Utils.getCitationContext(cm,true));  
			cm.setMetaData("goldSentiment", goldSentiment);
			citationMentions.add(cm);
			seen.add(surf);
		}
		return citationMentions;
	}

	/** 
	 * Compute evaluation metrics and write the output file.
	 * 
	 * @param pw A PrintWriter object associated with the output file
	 */
	public void writeEvaluation(PrintWriter pw) {
		int gTP = 0, gFP = 0;
		double gF1 = 0;
		for (CitationMention.Sentiment sent: CitationMention.Sentiment.values()){
			if (sent == CitationMention.Sentiment.NONE) continue;
			String a = sent.toString();
			int TP = (annoTP.get(a) == null ? 0 : annoTP.get(a).size());
			int FP = (annoFP.get(a) == null ? 0 : annoFP.get(a).size());
			int FN = (annoFN.get(a) == null ? 0 : annoFN.get(a).size());
			gTP += TP;
			gFP += FP;
			double precision = 0;
			double recall = 0;
			double f_measure = 0;
			if (TP+FP > 0) { precision = (double)TP/(TP+FP); }
			if (TP+FN > 0) { recall = (double)TP/(TP+FN); }
			if ((precision+recall) > 0) { 
				f_measure = (2*precision*recall)/(double)(precision+recall); 
			}
			gF1 += f_measure;

			pw.write(a + "\t" + (TP +FP) + "(" + TP + ")" + "\t" + (TP + FN) + "(" + TP + ")"
					+ "\t" + String.format("%1.4f", precision)
					+ "\t" + String.format("%1.4f", recall)
					+ "\t" + String.format("%1.4f", f_measure)); 		
			pw.write("\n");

			System.out.println(a + "\t" + (TP +FP) + "(" + TP + ")" + "\t" + (TP + FN) + "(" + TP + ")"
					+ "\t" + String.format("%1.4f", precision)
					+ "\t" + String.format("%1.4f", recall)
					+ "\t" + String.format("%1.4f", f_measure)); 		
		}

		double accuracy = 0;
		double macrof1 = 0;

		if (gTP+gFP > 0) { accuracy = (double)gTP/(gTP+gFP); }
		macrof1 = (double)gF1/3; 


		pw.write("Accuracy: " + String.format("%1.4f", accuracy)); pw.write("\n");
		pw.write("Macro-F1: " + String.format("%1.4f", macrof1)); pw.write("\n");
		pw.write("\n");

		System.out.println("Accuracy: " + String.format("%1.4f", accuracy)); 
		System.out.println("Macro-F1: " + String.format("%1.4f", macrof1)); 
	}

	public static void main(String[] argv) throws Exception {
		Properties props = FileUtils.loadPropertiesFromFile("citation.properties");
		String outfile = argv[0];
		RuleBasedSentimentEval eval = new RuleBasedSentimentEval(props);
		String trainDir = props.getProperty("sentimentTrainDirectory");
		String testDir = props.getProperty("sentimentAllDirectory");
		PrintWriter pw = new PrintWriter(outfile);
		eval.processDir(testDir,props,pw);
		eval.writeEvaluation(pw);
		pw.flush();
		pw.close();
	}
}
