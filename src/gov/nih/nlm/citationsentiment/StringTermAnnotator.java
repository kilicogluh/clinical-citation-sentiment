package gov.nih.nlm.citationsentiment;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.MultiWord;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.Word;
import gov.nih.nlm.ling.process.TermAnnotator;
import gov.nih.nlm.ling.sem.Concept;
import gov.nih.nlm.ling.sem.Ontology;
import gov.nih.nlm.ling.sem.SemanticItemFactory;

/**
 * An annotator for strings from a dictionary.
 * 
 * @author Halil Kilicoglu
 *
 */
public class StringTermAnnotator implements TermAnnotator {
	private static Logger log = Logger.getLogger(StringTermAnnotator.class.getName());	
	
	public static final Comparator<String> LENGTH_ORDER = 
	        new Comparator<String>() {
			public int compare(String s1, String s2) {
				if (s1.length() > s2.length()) return -1;
				else if (s2.length() > s1.length()) return 1;
				return s1.toLowerCase().compareTo(s2.toLowerCase());
			}	
		};
		
	private Map<String,String> dictionaryItems;
	private boolean allowMultipleAnnotations = true;
	private boolean postHyphenMatch = true;
		
	public StringTermAnnotator() {}
	
	/**
	 * Constructs an <code>StringTermAnnotator</code> with a list of simple dictionary terms
	 * 
	 * @param indicators	the set of indicators
	 */
	public StringTermAnnotator(Map<String,String> dictionaryItems) {
		this.dictionaryItems = dictionaryItems;
	}
	
	public Map<String,String> getDictionaryItems() {
		return dictionaryItems;
	}

	public void setDictionaryItems(Map<String,String> dictionaryItems) {
		this.dictionaryItems = dictionaryItems;
	}
	
	/**
	 * Returns whether multiple indicator annotations are allowed over the same text.
	 * 
	 * @return true if this variable has been set.
	 */
	public boolean allowMultipleAnnotations() {
		return allowMultipleAnnotations;
	}

	public void setAllowMultipleAnnotations(boolean allowMultipleAnnotations) {
		this.allowMultipleAnnotations = allowMultipleAnnotations;
	}
	
	/**
	 * Returns whether, if the token is hyphenated, the post-hyphen substring is allowed to be an indicator
	 * 
	 * @return true if this variable has been set.
	 */
	public boolean postHyphenMatch() {
		return postHyphenMatch;
	}

	public void setPostHyphenMatch(boolean postHyphenMatch) {
		this.postHyphenMatch = postHyphenMatch;
	}

	@Override 
	public void annotate(Document document, Properties props,
			Map<SpanList,LinkedHashSet<Ontology>> annotations) {
		if (dictionaryItems == null)
			throw new IllegalStateException("No dictionary terms have been loaded for annotation.");
		Map<String,String> termMap = new HashMap<>(dictionaryItems);
		// Annotate the larger indicators first
		List<String> termList = new ArrayList<>(termMap.keySet());
		Collections.sort(termList,LENGTH_ORDER);
		for (String term: termList) {
			log.log(Level.FINE,"Annotating term: {0}", new Object[]{term.toString()});
			annotateTerm(document,term,termMap.get(term),annotations);
		}
	}

	/**
	 * Annotates a given <code>Document</code> with the loaded indicators and 
	 * creates the corresponding <code>Predicate</code> objects for the mentions, as well.
	 * 
	 * @param document	the document to annotate
	 * @param props		the properties to use for annotation
	 */
	public void annotateTerms(Document document, Properties props) {
		Map<SpanList,LinkedHashSet<Ontology>> map = new LinkedHashMap<>();
		annotate(document,props,map);
		SemanticItemFactory sif = document.getSemanticItemFactory();
		for (SpanList sp: map.keySet()) {
			List<Word> words = document.getWordsInSpan(sp);
			SpanList headSpan = null;
			if (words.size() > 1) 
				headSpan = MultiWord.findHeadFromCategory(words).getSpan();
			else 
				headSpan = sp;
			LinkedHashSet<Ontology> inds = map.get(sp);
			for (Ontology ont: inds) {
				Concept conc = (Concept)ont;
				if (conc == null) {
					sif.newEntity(document, sp, headSpan,null);
				} else {
					String type = conc.getSemtypes().iterator().next();
					LinkedHashSet<Concept> concs = new LinkedHashSet<Concept>();
					concs.add(conc);
					sif.newEntity(document, sp, headSpan, type, concs, conc);
				}
			}
		}
	}
	
	/**
	 * Annotates terms specified as simple lowercase strings. If the word being examined is hyphenated, it attempts
	 * to annotate the indicator on part of the word, as well.
	 * 
	 * @param document  	the document
	 * @param term				term to annotate
	 * 	@param term				term type
	 * @param annotations	the updated annotations list
	 */
	public void annotateTerm(Document document, String term, String type, Map<SpanList, LinkedHashSet<Ontology>> annotations) {
		if (document.getText() == null) return;
		Pattern p = Pattern.compile("\\b" + Pattern.quote(term.toLowerCase()) + "(\\b|\\p{Punct})");
		String lw = document.getText().toLowerCase();
		Matcher m = p.matcher(lw);
		LinkedHashSet<String> types = new LinkedHashSet<>();
		types.add(type);
		while (m.find()) {
			SpanList sp = new SpanList(m.start(),m.end());
			Concept conc = new Concept("",term,types);
			LinkedHashSet<Ontology> concs = new LinkedHashSet<Ontology>();
			concs.add(conc);
			annotations.put(sp, concs);
		}
	}
}
