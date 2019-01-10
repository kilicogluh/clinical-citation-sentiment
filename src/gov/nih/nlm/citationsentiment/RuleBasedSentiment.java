package gov.nih.nlm.citationsentiment;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import gov.nih.nlm.ling.core.ContiguousLexeme;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.MultiWordLexeme;
import gov.nih.nlm.ling.core.Sentence;
import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.core.WordLexeme;
import gov.nih.nlm.ling.process.ComponentLoader;
import gov.nih.nlm.ling.process.SentenceSegmenter;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.Indicator;
import gov.nih.nlm.ling.sem.Predicate;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.sem.Sense;
import gov.nih.nlm.ling.sem.Term;
import gov.nih.nlm.ling.util.FileUtils;
import gov.nih.nlm.ling.wrappers.CoreNLPWrapper;

/**
 * Class to compute citation sentiment.
 * 
 * @author Halil Kilicoglu
 *
 */

public class RuleBasedSentiment {
	private static Logger log = Logger.getLogger(RuleBasedSentiment.class.getName());	

	public static RuleBasedSentiment instance; 

	//  private static Map<Class<? extends SemanticItem>,List<String>> annTypes;
	//  private static XMLReader xmlReader;

	private static Map<String,String> lexLines = new HashMap<>();
	private static Map<String,Double> sentimentScores = new HashMap<>();
	private static LinkedHashSet<Indicator>  sentimentTerms = new LinkedHashSet<>();
	private static Map<String,String> negTerms = new HashMap<>();

	private static Pattern CIT_PATTERN = Pattern.compile("<cit id=\"(.+?)\">(.+?)<\\/cit>");

	/**
	 * Creates an instance and initializes the dictionaries. 
	 * 
	 * @param props Properties to define dictionary locations ,etc.
	 * @return
	 */
	public static RuleBasedSentiment getInstance(Properties props)  {
		if (instance == null) {
			instance = new RuleBasedSentiment(props);
		}
		return instance;
	}
	
	private RuleBasedSentiment(Properties props)   {
		//	  xmlReader = getXMLReader();
		//	  annTypes = Utils.getAnnotationTypes();
		this(props, props.getProperty("scoreFile"));
	}
	
	private RuleBasedSentiment(Properties props, String filename)   {
		//	  xmlReader = getXMLReader();
		//	  annTypes = Utils.getAnnotationTypes();
		sentimentScores = loadSentimentScores(filename);
		sentimentTerms = loadSentimentTerms();
		try {
			negTerms = Utils.loadTermDictionary(props.getProperty("termDictionary"),Arrays.asList("NEGATION"));
		} catch (IOException ioe) {
			log.severe("Unable to load negation terms from file " + props.getProperty("termDictionary") + ". The program may not work as expected.");
		}
	}

	/*	private  XMLReader getXMLReader() {
		XMLReader reader = Utils.getXMLReader();
		reader.addAnnotationReader(Predicate.class, new XMLPredicateReader());
		return reader;
	}*/

	/**
	 * Loads sentiment dictionary scores from a file. Returns a map keyed by the lemma list 
	 * of the sentiment expression and with its precomputed score as the value.
	 * 
	 * @param filename	The name of the file.
	 * @return
	 */
	private Map<String,Double> loadSentimentScores(String filename)  {
		Map<String,Double> scs = new HashMap<>();
		try {
			List<String> lines = FileUtils.linesFromFile(filename, "UTF-8");
			for (String l: lines) {
				String[] els = l.split("[\\t]");
				String lex = els[3];
				if (els[2].equalsIgnoreCase("no")) continue;
				double sc = Double.parseDouble(els[7]);
				if (sc == 0.0) continue;
				scs.put(lex, sc);
				lexLines.put(lex,l);
			}
			log.info("Loaded " + scs.size() + " scored triggers.");

		} catch (IOException ioe) {
			log.severe("Unable to load scored triggers from " +  filename +".. Will terminate.");
			System.exit(1);
		}
		return scs;
	}

	private LinkedHashSet<Indicator> loadSentimentTerms()  {
		LinkedHashSet<Indicator> allIndicators = new LinkedHashSet<>();
		for (String a: sentimentScores.keySet()) {
			//			if (a.equals("NO_LEMMA")) continue;
			String[] as = a.split("[ ]+");
			String[] els = lexLines.get(a).split("\\t");
			List<WordLexeme> lexemes = new ArrayList<>(as.length);
			for (int i=0; i < as.length; i++) {
				String inds = as[i];
				String lemma = inds.substring(0, inds.indexOf("("));
				String cat = inds.substring(inds.indexOf("(") +1,inds.indexOf(")"));
				WordLexeme lex = new WordLexeme(lemma,cat);
				lexemes.add(lex);
			}
			ContiguousLexeme rl = null;
			if (lexemes.size() == 1) {
				rl = lexemes.get(0);
			} else {
				rl = new MultiWordLexeme(lexemes);
			}		
			Indicator ind = new Indicator(els[0],Arrays.asList(rl),false,Arrays.asList(new Sense(els[1])));
			allIndicators.add(ind);
		}
		for (Indicator a: allIndicators) {
			log.fine("Loaded trigger:" + a.toString());
		}
		return allIndicators;
	}


	/** 
	 * Processes all citation mentions in a document.
	 * 
	 * @param doc Document to process
	 * @return	a String representation of the results (one line per citation mention)
	 */
	public String processMentions(Document doc) {
		LinkedHashSet<SemanticItem> cms = Document.getSemanticItemsByClass(doc, CitationMention.class);
		StringBuilder buf = new StringBuilder();
		for (SemanticItem s: cms) {
			CitationMention cm = (CitationMention)s;
			String str = processMention(cm);
			buf.append(str); buf.append("\n");
		}
		return buf.toString();
	}

	/**
	 * Processes a single citation mention.
	 * 
	 * @param mention	Citation mention to process
	 * @return	a single line String representation of the result (<code>DocID|MentionID|MentionText|Prediction|Score|Triggers</code>)
	 */
	public String processMention(CitationMention mention) {
		Map<String,Double> scoreMap = calculateScoreMap(mention);
		double cumScore = 0.0;
		for (String c: scoreMap.keySet()) {
			cumScore += scoreMap.get(c);
		}
		String predict = predict(scoreMap);
		StringBuffer scoreBuf = new StringBuffer();
		for (String t: scoreMap.keySet()) {
			scoreBuf.append(t + "(" + scoreMap.get(t) + "),");
		}
		String scoreStr = scoreBuf.toString();
		return (mention.getDocument().getId() + "|" + mention.getId() + "|" + mention.getText() + "|" + predict + "|" +  cumScore + "|" + 
				(scoreStr.equals("") ? "" : scoreStr.substring(0,scoreStr.length()-1)));
	}

	/**
	 * Reads and annotates a single text file. The file is expected to consists of one sentence per line.
	 * Each citation mention should be surrounded by "cit" tags. (e.g. <pre>{@code <cit id="C1">[1-5]</cit>}</pre>)
	 * 
	 * @param filename	The filename to process
	 * @param props	Properties to use in annotation
	 * @return a Document object with all the relevant terms and citation mentions annotated, null if the processing fails
	 */
	public Document  preProcessTextFile(String filename,Properties props)  {
		Document doc = null;
		try {
			String text = FileUtils.stringFromFile(filename, "UTF-8");
			doc = preProcessString(filename,text,props);

		} catch (IOException ie) {
			log.severe("Unable to process input file " + filename);
			ie.printStackTrace();
			System.exit(1);
		} 
		return doc;
	}
	
	public Document  preProcessString(String id, String input,Properties props)  {
		Document doc = null;
		try {
			SentenceSegmenter segmenter = ComponentLoader.getSentenceSegmenter(props);
			CoreNLPWrapper.getInstance(props);
			LinkedHashMap<String,SpanList> citSpans = identifyCitationSpans(input);
			String textOnly =input.replaceAll("<(.+?)>", "").trim();
			doc = new Document(id,textOnly);
			CitationFactory sf = new CitationFactory(doc,new HashMap<>());
			doc.setSemanticItemFactory(sf);
			List<Sentence> sentences = new ArrayList<>();
			segmenter.segment(doc.getText(), sentences);
			doc.setSentences(sentences);
			for (int i=0; i < doc.getSentences().size(); i++) {
				Sentence sent = doc.getSentences().get(i);
				sent.setDocument(doc);
				// create word list, pos, lemma info
				CoreNLPWrapper.coreNLP(sent);
			}

			for (String st: citSpans.keySet()) {
				SpanList sp = citSpans.get(st);
				CitationMention m = sf.newCitationMention(doc, "CitationMention", sp, sp, doc.getText().substring(sp.getBegin(), sp.getEnd()));
				m.setContext(Arrays.asList(doc.getSubsumingSentence(sp.asSingleSpan()).getSpan()));
				m.setId(st);
			}
			annotateTerms(doc,props);
		} catch (Exception e) {
				log.warning("Unable to segment sentences.");
				e.printStackTrace();
			}
		return doc;
	}

	private LinkedHashMap<String,SpanList> identifyCitationSpans(String text) {
		LinkedHashMap<String,SpanList> spans = new LinkedHashMap<>();
		int ind= 0;
		StringBuilder buf = new StringBuilder();
		Matcher m = CIT_PATTERN.matcher(text);
		while (m.find()) {
			String pre = text.substring(ind,m.start());
			int citBeg = buf.toString().length() + pre.length();
			String citId = m.group(1);
			buf.append(pre);
			buf.append(m.group(2));
			SpanList sp = new SpanList(citBeg,citBeg + m.group(2).length());
			ind = m.end();
			spans.put(citId, sp);
		}
		return spans;
	}

	/**
	 * Annotates a document with sentiment-related terms.
	 * 
	 * @param doc	The document to annotate
	 * @param properties		Properties relevant for annotation
	 */
	public void annotateTerms(Document doc, Properties properties)  {
		Utils.annotateIndicators(doc,sentimentTerms,properties);
		Utils.annotateStrings(doc, negTerms, properties);
		Utils.removeSubsumedTerms(doc);
	}

	/**
	 * Computes a score map for a citation mention. 
	 * 
	 * @param mention	The citation mention 
	 * @return	a map where keys are sentiment terms found and the values are scores associated with them (taking negation into account).
	 */
	public  Map<String,Double> calculateScoreMap(CitationMention mention)   {
		List<Span> context = mention.getContext();
		Document doc = mention.getDocument();
		/*	  LinkedHashSet<SemanticItem> preds = Document.getSemanticItemsByClass(doc, Predicate.class);
	  if (preds.size() == 0) {
		  Utils.annotateIndicators(doc,sentimentTerms,properties);
	  }*/
		SurfaceElement surf = mention.getSurfaceElement();
		int contextSize = 0;
		int index = 0;
		for (Span sp : context) {
			List<SurfaceElement> surfs = doc.getSurfaceElementsInSpan(sp);
			if (surfs.contains(mention.getSurfaceElement())) {
				contextSize = surfs.size();
				index = context.indexOf(sp);
				break;
			}
		}
		Map<String,Double> scoreMap = new HashMap<>();
		for (Span sp: context) {
			LinkedHashSet<SemanticItem> sems = Document.getSemanticItemsByClassSpan(doc, Term.class, new SpanList(sp), false);
			for (SemanticItem sem : sems) {
				if (sem.getType().equals("NEGATION")) continue;
				if (sem instanceof Entity || sem instanceof CitationMention) continue;
				Predicate pred = (Predicate)sem;
				SurfaceElement prs = pred.getSurfaceElement();
				SurfaceElement prev =prs.getSentence().getPrecedingSurfaceElement(prs);
				if (prev == null && pred.getType().equals("discourse")) continue;
				Indicator ind = pred.getIndicator();
				String key = ind.getLexeme().toString().replaceAll("_", " ");
				double score = sentimentScores.get(key);
				int intrv = contextSize;
				if (context.indexOf(sp) == index) 
					intrv = doc.getInterveningSurfaceElements(prs, surf).size();
				double factor = (double)Math.log(2)/Math.log(intrv+2);
				boolean neg = false;
				if (prev != null) {
					LinkedHashSet<SemanticItem> negs =prev.filterByEntities();
					for (SemanticItem n: negs) {
						if (n.getType().equals("NEGATION")) {
							neg = true;
							break;
						}
					}
				} 
				if (!neg) {
					SurfaceElement prevprev =prs.getSentence().getPrecedingSurfaceElement(prev);
					if (prevprev != null) {
						LinkedHashSet<SemanticItem> negs =prevprev.filterByEntities();
						for (SemanticItem n: negs) {
							if (n.getType().equals("NEGATION")) {
								neg = true;
								break;
							}
						} 
					}
				}
				if (score > 0) {
					if (!neg) scoreMap.put(key, score);
					else scoreMap.put(key, -score);
				} else {
					if (!neg) scoreMap.put(key, score*factor);
					else scoreMap.put(key, -score*factor);
				}
			}
		}
		return scoreMap;
	}

	/**
	 * Returns the prediction based on the computed score map.
	 * 
	 * @param map	The score map 
	 * @return "NEUTRAL", "NEGATIVE" or "POSITIVE"
	 */
	public String predict(Map<String,Double> map) {
		double cumScore = 0.0;
		for (String c: map.keySet()) {
			cumScore += map.get(c);
		}
		String predict = "NEUTRAL";
		if (cumScore > 1) predict = "POSITIVE";
		else if (cumScore <-0.1) predict = "NEGATIVE";
		return predict;
	}

	/**
	 * Command line (intended entry).
	 */
	public static void main(String[] argv) throws Exception {
		Properties props = FileUtils.loadPropertiesFromFile("citation.properties");
		String infile = argv[0];
		String outfile = argv[1];
		RuleBasedSentiment instance = RuleBasedSentiment.getInstance(props);

		Document doc = instance.preProcessTextFile(infile,props);
		if (doc != null) {
			String out =  instance.processMentions(doc);
			System.out.println(out);
			PrintWriter pw = new PrintWriter(outfile);
			pw.println(out);
			pw.flush();
			pw.close();
		}
	}
}
