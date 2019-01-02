package gov.nih.nlm.citationsentiment;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.Logger;

import gov.nih.nlm.ling.core.ContiguousLexeme;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.MultiWordLexeme;
import gov.nih.nlm.ling.core.Sentence;
import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.core.WordLexeme;
import gov.nih.nlm.ling.io.XMLReader;
import gov.nih.nlm.ling.process.ComponentLoader;
import gov.nih.nlm.ling.process.IndicatorAnnotator;
import gov.nih.nlm.ling.process.TermAnnotator;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.Indicator;
import gov.nih.nlm.ling.sem.Predicate;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.sem.Sense;
import gov.nih.nlm.ling.sem.Term;
import gov.nih.nlm.ling.util.FileUtils;


/** 
 * Utility methods to use in processing citation mentions.
 * 
 * @author Halil Kilicoglu
 *
 */
public class Utils {
	private static Logger log = Logger.getLogger(Utils.class.getName());
	
	public static Map<Class<? extends SemanticItem>,List<String>> getAnnotationTypes() {
		Map<Class<? extends SemanticItem>,List<String>> annTypes = new HashMap<Class<? extends SemanticItem>,List<String>>();
		annTypes.put(CitationMention.class,Arrays.asList("CitationMention"));
		return annTypes;
	}
	
	public static XMLReader getXMLReader() {
		XMLReader reader = new XMLReader();
		reader.addAnnotationReader(CitationMention.class, new XMLCitationMentionReader());
		return reader;
	}
	
/*	public static Map<Class<? extends SemanticItem>,List<String>> getAnnotationTypesMML() {
		Map<Class<? extends SemanticItem>,List<String>> annTypes = new HashMap<Class<? extends SemanticItem>,List<String>>();
		annTypes.put(CitationMention.class,Constants.ANNOTATION_TYPES);
		annTypes.put(Entity.class, Constants.SEMREP_ENTITY_ABRRVS);
		return annTypes;
	}*/
	
/*	public static XMLReader getXMLReaderMML() {
		XMLReader reader = new XMLReader();
		reader.addAnnotationReader(CitationMention.class, new XMLCitationMentionReader());
		reader.addAnnotationReader(Entity.class,new XMLEntityReader());
		return reader;
	}*/
	
	public static Map<String,Map<Integer,List<Integer>>> readFolds(String dir) throws IOException {
		Map<String,Map<Integer,List<Integer>>> folds = new HashMap<>();
		List<String> files = FileUtils.listFiles(dir, false, "txt");
		Collections.sort(files);
		for (String f: files) {
			if (f.contains("test") == false &&  f.contains( "train") == false) continue;
			String ff = f.substring(f.lastIndexOf(File.separator)+1);

			boolean test = ff.startsWith("test");
			int fold = (test? Integer.parseInt(ff.substring(4,5)) : Integer.parseInt(ff.substring(5,6)));
			System.out.println("FILE: " + f + "|" + ff + "|" + fold + "|" +  (test ? "test" : "train"));
			List<String> lines = FileUtils.linesFromFile(f, "UTF-8");
			List<Integer> allIds = new ArrayList<>();
			for (String l: lines) {
				String[] els = l.split("[\\t]");
				int id = Integer.parseInt(els[0]);
				allIds.add(id);
			}
			if (test) {
				Map<Integer,List<Integer>> testFolds = folds.get("test");
				if (testFolds == null) {
					testFolds = new HashMap<>();
				}
				testFolds.put(fold, allIds);
				folds.put("test", testFolds);
			}
			else {
				Map<Integer,List<Integer>> trainFolds = folds.get("train");
				if (trainFolds == null) {
					trainFolds = new HashMap<>();
				}
				trainFolds.put(fold, allIds);
				folds.put("train", trainFolds);
			}
		}
		return folds;
	}
	
	/**
	 * Annotates a given document with a set of provided indicators (triggers) based on lemmas.
	 * {@link gov.nlm.nih.gov.process.IndicatorAnnotator} class is used. 
	 * 
	 * @param doc		Document to annotate
	 * @param indicators	Set of indicators specified with their lemmas
	 * @param props	Additional properties to pass to the annotator
	 */
	public static void annotateIndicators(Document doc, LinkedHashSet<Indicator> indicators, Properties props)  {
		props.setProperty("termAnnotators","gov.nih.nlm.ling.process.IndicatorAnnotator");
//		props.put("ignorePOSforIndicators", "true");
		try {
			List<TermAnnotator> termAnnotators = ComponentLoader.getTermAnnotators(props);
			for (TermAnnotator annotator : termAnnotators) {
				if (annotator instanceof IndicatorAnnotator) {
					((IndicatorAnnotator)annotator).setIndicators(indicators);
					String ignore = props.getProperty("ignorePOSforIndicators");
					boolean ignorePOS = Boolean.parseBoolean(ignore == null ? "false" : ignore);
					((IndicatorAnnotator)annotator).setIgnorePOS(ignorePOS);
					((IndicatorAnnotator)annotator).annotateIndicators(doc, props);
				}
			}
		} catch (Exception ie) {
			log.severe("Unable to instantiate the trigger annotator.");
			ie.printStackTrace();
		}
	}
	
	/**
	 * Annotates a given document with a set of strings from a dictionary.
	 * {@link StringTermAnnotator} is used. 
	 * 
	 * @param doc		Document to annotate
	 * @param dictionaryItems		Dictionary of strings to annotate with
	 * @param props	Additional properties to pass to the annotator
	 */
	public static void annotateStrings(Document doc, Map<String,String>dictionaryItems, Properties props)  {
		props.setProperty("termAnnotators","gov.nih.nlm.citationsentiment.StringTermAnnotator");
		try {
		List<TermAnnotator> termAnnotators = ComponentLoader.getTermAnnotators(props);
			for (TermAnnotator annotator : termAnnotators) {
				if (annotator instanceof StringTermAnnotator) {
					((StringTermAnnotator)annotator).setDictionaryItems(dictionaryItems);
					((StringTermAnnotator)annotator).annotateTerms(doc,props);
				}
			}
		} catch (Exception ie) {
			log.severe("Unable to instantiate string annotator.");
			ie.printStackTrace();
		}
	}

	/**
	 * Removes predicates subsumed by other terms from the document.
	 * 
	 * @param doc	Document to process
	 */
	public static void removeSubsumedPredicates(Document doc) {
		List<SemanticItem> toRemove = new ArrayList<SemanticItem>();
		LinkedHashSet<SemanticItem> terms = Document.getSemanticItemsByClass(doc, Predicate.class);
		for (SemanticItem term : terms) {
			Predicate t = (Predicate)term;
//			Indicator ind = t.getIndicator();
			SurfaceElement su = t.getSurfaceElement();
			LinkedHashSet<SemanticItem> suTerms = su.filterByPredicates();
			for (SemanticItem suT: suTerms) {
				if (suT.equals(t) || toRemove.contains(suT)) continue;
//				if (indicators.contains(ind) == false) continue;
				if (SpanList.subsume(suT.getSpan(), t.getSpan()) && suT.getSpan().length() > t.getSpan().length()) {
					toRemove.add(t);
					break;
				}
			}
		}
		for (SemanticItem rem: toRemove) {
			log.finest("Removing subsumed predicate " + rem.toShortString());
			doc.removeSemanticItem(rem);
		}
	}
	
	/**
	 * Removes all terms subsumed by others from the given document.
	 * 
	 * @param doc	Document to process
	 */
	public static void removeSubsumedTerms(Document doc) {
		List<SemanticItem> toRemove = new ArrayList<SemanticItem>();
		LinkedHashSet<SemanticItem> terms = Document.getSemanticItemsByClass(doc, Term.class);
		for (SemanticItem term : terms) {
			Term t = (Term)term;
			SurfaceElement su = t.getSurfaceElement();
			LinkedHashSet<SemanticItem> suTerms = su.filterByTerms();
			for (SemanticItem suT: suTerms) {
				if (suT.equals(t) || toRemove.contains(suT)) continue;
//				if (indicators.contains(ind) == false) continue;
				if (SpanList.subsume(suT.getSpan(), t.getSpan()) && suT.getSpan().length() > t.getSpan().length()) {
					toRemove.add(t);
					break;
				}
			}
		}
		for (SemanticItem rem: toRemove) {
			log.finest("Removing subsumed term " + rem.toShortString());
			doc.removeSemanticItem(rem);
		}
	}
	
	/**
	 * Removes all entities subsumed by other terms from the given document.
	 * 
	 * @param doc	Document to process
	 */
	public static void removeSubsumedEntities(Document doc) {
		List<SemanticItem> toRemove = new ArrayList<SemanticItem>();
		LinkedHashSet<SemanticItem> terms = Document.getSemanticItemsByClass(doc, Entity.class);
		for (SemanticItem term : terms) {
			Entity t = (Entity)term;
			SurfaceElement su = t.getSurfaceElement();
			LinkedHashSet<SemanticItem> suTerms = su.filterByEntities();
			for (SemanticItem suT: suTerms) {
				if (suT.equals(t) || toRemove.contains(suT)) continue;
//				if (indicators.contains(ind) == false) continue;
				if (SpanList.subsume(suT.getSpan(), t.getSpan()) && suT.getSpan().length() > t.getSpan().length()) {
					toRemove.add(t);
					break;
				}
			}
		}
		for (SemanticItem rem: toRemove) {
			log.finest("Removing subsumed entity " + rem.toShortString());
			doc.removeSemanticItem(rem);
		}
	}
	
	/**
	 * Returns whether a given sentence starts with a contrastive marker, such as <i>although</i>, <i>however</i>.
	 * The sentence is expected to have been annotated with all relevant terms already (CONTRAST type  is relevant 
	 * in this case).
	 * 
	 * @param sentence		Sentence to assess
	 */
	public static boolean sentenceInitialContrastive(Sentence sentence) {
		if (sentence.getSurfaceElements() == null) return false;
		SurfaceElement f = sentence.getSurfaceElements().get(0);
		LinkedHashSet<SemanticItem> es = f.filterByEntities();
		if (es == null || es.size() ==0) return false;
		for (SemanticItem e: es) {
			if (e.getType().equals("CONTRAST")) return true;
		}
		return false;
	}
	
	/**
	 * Returns the context for a given citation mention. Currently, it is either the current sentence only 
	 * or the current sentence plus the next, unless the next sentence starts with a contrastive marker.
	 * 
	 * @param mention		Mention in question
	 * @param currentSentenceOnly	Whether to limit to the current sentence
	 * @return		A list of character offsets, which collectively define the context.
	 */
	  public static List<Span> getCitationContext(CitationMention mention, boolean currentSentenceOnly) {
		  List<Span> context = new ArrayList<>();
		  Sentence sent = mention.getSurfaceElement().getSentence();
		  Document doc = mention.getDocument();
		  context.add(sent.getSpan());
		  if (currentSentenceOnly) return context;
		  int ind = doc.getSentences().indexOf(sent);
		  if (ind == doc.getSentences().size() -1) return context;
		  Sentence next = doc.getSentences().get(ind+1);
		  Span nextSpan = next.getSpan();
		  LinkedHashSet<SemanticItem> nextMentions = Document.getSemanticItemsByClassSpan(doc, CitationMention.class, new SpanList(nextSpan), false);
		  if (nextMentions == null || nextMentions.size() == 0) {
			  if (sentenceInitialContrastive(next)) {
				  context.add(nextSpan);
			  }
		  } 
		  return context;
	  }
 
	  /**
	   * Returns clause level contexts, if relevant. If the mention is the only one in a sentence, 
	   * the context is simply the sentence. If not, a mention's context is the span between the previous 
	   * citation mention and the current mention.  
	   * 
	   * 
	   * @param m
	   * @return
	   */
	  public static List<Span> getCitationContextSubSentential(CitationMention m) {
		  List<Span> context = new ArrayList<>();
		  SurfaceElement surf = m.getSurfaceElement();
		  Sentence sent =surf.getSentence();
		  Document doc = m.getDocument();
		  Span u = new Span(sent.getSpan().getBegin(),m.getSpan().getBegin());
		  LinkedHashSet<SemanticItem> prevMentions = Document.getSemanticItemsByClassSpan(doc, CitationMention.class, new SpanList(u), false);
		  LinkedHashSet<SemanticItem> allMentions = Document.getSemanticItemsByClassSpan(doc, CitationMention.class, new SpanList(sent.getSpan()), false);
		  LinkedHashSet<SemanticItem> sameMention = Document.getSemanticItemsByClassSpan(doc, CitationMention.class, m.getSpan(), false);
		  // only citation in the sentence
		  if (allMentions.size() == 1 || (sameMention.size() == allMentions.size())) {
			  context.add(sent.getSpan());
			  return context;
		  }
		  // span up to the mention only
		  if (prevMentions.size() == 0) {
			  context.add(new Span(sent.getSpan().getBegin(),m.getSpan().getBegin()));
		  }
		  // span from the previous mention up to the current mention only
		  else {
			  SpanList closest = null;
			  for (SemanticItem mm: prevMentions) {
				  if (closest == null || SpanList.atLeft(closest,mm.getSpan())) closest = mm.getSpan();
			  }
			  context.add(new Span(closest.getEnd(),m.getSpan().getBegin()));
		  }
		  int ind = doc.getSentences().indexOf(sent);
		  if (ind == doc.getSentences().size() -1) return context;
		  Sentence next = doc.getSentences().get(ind+1);
		  Span nextSpan = next.getSpan();
		  LinkedHashSet<SemanticItem> nextMentions = Document.getSemanticItemsByClassSpan(doc, CitationMention.class, new SpanList(nextSpan), false);
		  if (nextMentions == null || nextMentions.size() == 0) {
			  if (sentenceInitialContrastive(next)) {
				  context.add(nextSpan);
			  }
		  } 
		  return context; 
	  }
	 
	  /**
	   * Loads a dictionary of sentiment terms (lemma-based).
	   * 
	   * @param filename	The dictionary file
	   * @return	A set of indicators 
	   * @throws IOException
	   */
	public static  LinkedHashSet<Indicator> loadSentimentDictionary(String filename) throws IOException {
			 LinkedHashSet<Indicator>dictionary = new LinkedHashSet<>();
		final List<String> lines = FileUtils.linesFromFile(filename, "UTF8");
		for (String l: lines) {
//			System.out.println("LINE:" + l);
			if (l.startsWith("#")) continue;
			String[] els = l.split("\t");
			String text = els[0];
			String pos = els[1];
			String cat = els[2];
			String[] textsubs = text.split("[ ]+");
			String[] possubs = pos.split("[ ]+");
			List<WordLexeme> lexs = new ArrayList<>();
			for (int i=0; i < textsubs.length; i++) {
				lexs.add(new WordLexeme(textsubs[i],possubs[i]));
			}
			List<ContiguousLexeme> indlexs = new ArrayList<>();
			indlexs.add(new MultiWordLexeme(lexs));
			
			Indicator ind = new Indicator(text,indlexs,true,Arrays.asList(new Sense(cat)));
			dictionary.add(ind);
		}
		return dictionary;
	}
	
	/**
	 * Loads a simple term dictionary.
	 * 
	 * @param filename	The dictionary file
	 * @return
	 * @throws IOException
	 */
	public static  Map<String,String> loadTermDictionary(String filename ) throws IOException {
		Map<String,String> dictionary = new HashMap<>();
		final List<String> lines = FileUtils.linesFromFile(filename, "UTF8");
		for (String l: lines) {
			//		System.out.println("LINE:" + l);
			if (l.startsWith("#")) continue;
			String[] els = l.split("\t");
			String text = els[0];
			String type = els[1];
			dictionary.put(text,type);
		}
		log.fine("Loaded " + dictionary.size() + " triggers from " + filename);
		return dictionary;
	}
	
	/**
	 * Loads only dictionary items with specific types.
	 * 
	 * @param filename	The dictionary file
	 * @param types		Semantic types to load
	 * @return
	 * @throws IOException
	 */
	public static  Map<String,String> loadTermDictionary(String filename, List<String> types ) throws IOException {
		Map<String,String> dictionary = new HashMap<>();
		final List<String> lines = FileUtils.linesFromFile(filename, "UTF8");
		for (String l: lines) {
			if (l.startsWith("#")) continue;
			String[] els = l.split("\t");
			String text = els[0];
			String type = els[1];
			if (types == null || types.contains(type)) 
				dictionary.put(text,type);
		}
		log.fine("Loaded " + dictionary.size() + " triggers from " + filename);
	return dictionary;
}
	
	/**
	 * Returns whether the surface unit is a citation mention.
	 * 
	 * @param surf	The surface unit in question
	 * @return
	 */
	public static boolean isCitationMention(SurfaceElement surf) {
		LinkedHashSet<SemanticItem> cits = surf.filterSemanticsByClass(CitationMention.class);
		return (cits != null && cits.size() > 0);
	}
	
	/** 
	 * Returns whether a citation mention should be treated as the current instance or not..
	 * 
	 * @param mention	The citation mention in question
	 * @param token		The citation token
	 * @return THISCITATION if the citation mention is associated with the token, OTHERCITATION if not, 
	 * 					null if token is not associated with any citation.
	 */
	public static String getCitationString(CitationMention mention, SurfaceElement token) {
		if (mention.getSurfaceElement().equals(token)) return "THISCITATION ";
		else if (isCitationMention(token)) return "OTHERCITATION ";
		return null;
	}
	
	/**
	 * Loads results of rule-based method from a file. Useful for extracting ML features.
	 * 
	 * @param filename	The file containing the rule-based method results
	 * 
	 * @return	a Map of IDs and citation sentiment values
	 */
	public static Map<String,String> loadRuleBasedResultsFromFile(String filename)  {
		Map<String,String> results = new HashMap<>();
		try {
		List<String> lines = FileUtils.linesFromFile(filename, "UTF-8");
		for (String l: lines) {
			String[] els = l.split("\\|");
			if (els.length < 5) continue;
			String a = els[4];
			String[] els1 = a.split("_");
			String id = els1[0];
			String cid = els1[1];
//			ruleBasedScores.put(id + "_" + cid, Double.parseDouble(els[2]));
			results.put(id+ "_" + cid, els[0]);
		}
		} catch (IOException ioe) {
			log.severe("Rule-based sentiment result file cannot be opened.");
		}
		return results;
	}
	
/*	public static LinkedHashSet<String> loadClinicalTermDictionary(String filename) throws IOException {
		List<String> termlist = new ArrayList<>();
		final List<String> lines = FileUtils.linesFromFile(filename, "UTF8");
		for (String l: lines) {
			if (l.startsWith("#")) continue;
			termlist.add(l.trim());
		}
		Collections.sort(termlist,new Comparator<String>()  {
			public int compare(String a, String b) {
				int al = a.length();
				int bl = b.length();
				if (al == bl) return a.compareTo(b);
				return (bl-al);
			}
		});
		return new LinkedHashSet<String>(termlist);
	}*/
	

	

}
