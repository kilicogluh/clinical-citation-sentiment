package gov.nih.nlm.citationsentiment.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeSet;
import java.util.logging.Logger;

import gov.nih.nlm.citationsentiment.CitationMention;
import gov.nih.nlm.citationsentiment.RuleBasedSentiment;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.Sentence;
import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.core.SynDependency;
import gov.nih.nlm.ling.core.Word;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.sem.Term;
import gov.nih.nlm.ml.feature.DoubleFeature;
import gov.nih.nlm.ml.feature.Feature;
import gov.nih.nlm.ml.feature.FeatureSet;
import gov.nih.nlm.ml.feature.StringFeature;
import gov.nih.nlm.ml.feature.StringSetFeature;
import gov.nih.nlm.util.Strings;
import liblinear.FeatureNode;

public class Features<T extends CitationMention> extends FeatureSet<T> {
	private static Logger log = Logger.getLogger(Features.class.getName());	
	
	private  Map<String,String> usedTerms = null;
	
	public static final Comparator<FeatureNode> FEATURE_NODE_COMPARATOR =
			new Comparator<FeatureNode>() {
		public int compare(final FeatureNode f1, final FeatureNode f2) {
			return f1.index - f2.index;
		}
	};
	
	public void setUsedTerms(Map<String,String> terms) {
		this.usedTerms = terms;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public Set<Feature<T,?>> getFeatures() {
		final Set<Feature<T,?>> features = newSet();
		for (final Type type : Type.values()) {
			for (int i=1; i<= 5; i++) {
				features.addAll(getFeatures(type,i));
			}
		}
		return features;
	}

	public Set<Feature<T,?>> getFeatures(final Type type, final int window) {
		final Set<Feature<T,?>> features = newSet();
		return features;
	}

	public enum Type {
		NORMAL {
			public String convert(final SurfaceElement token) {
				return token.getText();
			}
		},

		UNCASED {
			@Override
			public String convert(final SurfaceElement token) {
				return token.getText().toLowerCase();
			}
		},

		STEMMED {
			@Override
			public String convert(final SurfaceElement token) {
				return token.getLemma().toLowerCase();
			}
		},

		UNCASED_WORDS {
			@Override
			public String convert(final SurfaceElement token) {
				if (token instanceof Word) {
					final String rawString = token.getText().toLowerCase();
					if (Strings.containsLetter(rawString)) return rawString;
					return null;
				} else {
					return token.getText().toLowerCase();
				}
			}
		},
		
		STEMMED_WORDS {
			@Override
			public String convert(final SurfaceElement token) {
				final String rawString = token.getLemma().toLowerCase();
				if (Strings.containsLetter(rawString)) return rawString;
				return null;
			}
		},
		
		POS {
			@Override
			public String convert(final SurfaceElement token) {
				return token.getPos();
			}
		},
		
		UNCASED_POS {
			@Override
			public String convert(final SurfaceElement token) {
					return token.getText().toLowerCase()  + "_" + token.getPos();
			}
		},
		
		STEMMED_POS {
			@Override
			public String convert(final SurfaceElement token) {
				final String rawString = token.getLemma().toLowerCase();
				return rawString + "_" + token.getPos();
			}
		},
		
		CAT {
			@Override
			public String convert(final SurfaceElement token) {
				return token.getCategory();
			}
		},
		
		STEMMED_CAT {
			@Override
			public String convert(final SurfaceElement token) {
				final String rawString = token.getLemma().toLowerCase();
				return rawString + "_" + token.getCategory();
			}
		};

		/**
		 * Converts the given {@link SurfaceElement} to a <code>String</code>.  A
		 * <code>null</code> value indicates the <var>token</var> should not be
		 * included.
		 */
		public abstract String convert(SurfaceElement token);
	}
	
	  public enum TokenType {		
			WORD {
				@Override
				public List<SurfaceElement> convert(final Sentence sentence) {
					return new ArrayList<SurfaceElement>(sentence.getWords());
				}
			},
			
			UNIT {
				@Override
				public List<SurfaceElement> convert(final Sentence sentence) {
					return sentence.getSurfaceElements();
				}
			};
			
			public abstract List<SurfaceElement> convert(Sentence sentence);
		  }
	
	  
/*	public static void surfaceElementNgrams(final List<SurfaceElement> surfaceElements, final CitationMention cm,
			final int n,
			final Type type,
			final Collection<String> grams) {
		for (int i = 0; i <= surfaceElements.size() - n; i++) {
			final List<String> words = new ArrayList<String>(n);
			for (int j = i; j < i + n; j++) {
				final SurfaceElement token = surfaceElements.get(j);
				String cs = getCitationString(cm,token);
				if (cs != null) {
					words.add(cs);
				}
				else {
					final String word = type.convert(token);
					if (word == null) {
						break;
					}
					else {
						words.add(word);
					}
				}
			}
			if (words.size() == n) {
				grams.add(Strings.join(words, "__"));
			}
		}
	}*/
	
/*	public static void ngrams(final List<Word> words, final CitationMention cm,
			final int n,
			final Type type,
			final Collection<String> grams) {
		for (int i = 0; i <= words.size() - n; i++) {
			final List<String> ngrams = new ArrayList<String>(n);
			for (int j = i; j < i + n; j++) {
				final Word token = words.get(j);
				String cs = getCitationString(cm,token);
				if (cs != null) {
					ngrams.add(cs);
					continue;
				}
				final String word = type.convert(token);
				if (word == null) {
					break;
				}
				else {
					ngrams.add(word);
				}
			}
			if (ngrams.size() == n) {
				grams.add(Strings.join(ngrams, "__"));
			}
		}
	}*/
	
	public static void stringNgrams(final List<String> words, 
			final int n,
			final Collection<String> grams) {
		for (int i = 0; i <= words.size() - n; i++) {
			final List<String> ngrams = new ArrayList<String>(n);
			for (int j = i; j < i + n; j++) {
				final String word = words.get(j);
				if (word == null) {
					break;
				}
				else {
					ngrams.add(word);
				}
			}
			if (ngrams.size() == n) {
				grams.add(Strings.join(ngrams, "__"));
			}
		}
	}
		
	private class ContextNGramFeature extends StringSetFeature<T> {
		private final int n;
		private final Type type;
		protected ContextNGramFeature(final String name,
				final int n,
				final Type type) {
			super(name);
			this.n = n;
			this.type = type;
		}
		@Override
		public Set<String> compute(final T span) {
			final List<Span> context = span.getContext();
			final Set<String> grams = new TreeSet<String>();
			List<String> tokens = new ArrayList<>();
			for (Span sp: context) {
				List<SurfaceElement> surfs = span.getDocument().getSurfaceElementsInSpan(sp);
				for (SurfaceElement surf: surfs) {
					if (isCitationMention(surf)) {
						String cs = getCitationString(span,surf);
						tokens.add(cs);
					} else {
						for (Word w: surf.toWordList()) {
							tokens.add(type.convert(w));
						}
					}			
				}
			}
			stringNgrams(tokens, n,grams);
			return grams;
		}
	}
	
	public class NegationNGramFeature extends StringSetFeature<T> {
		private final Type type;
		protected NegationNGramFeature(final String name,
				final Type type) {
			super(name);
			this.type = type;
		}
		@Override
		public Set<String> compute(final T cm) {
			List<SurfaceElement> negs = getNegationClues(cm);
			Set<String> strs = new HashSet<>();
			for (SurfaceElement neg: negs) {
				Sentence sent = neg.getSentence();
				int negind = sent.getSurfaceElements().indexOf(neg);
				int total = sent.getSurfaceElements().size();
				SurfaceElement next = null; SurfaceElement nextnext = null;
				if (negind < total -1) 
					next = sent.getSurfaceElements().get(negind+1);
				if (negind < total -2)
					nextnext = sent.getSurfaceElements().get(negind+2);
				if (next != null) {
					String cs = getCitationString(cm,next);
					if (cs != null) {
						strs.add("NOT_" + cs);
					}
					else {
						String c = type.convert(next);
						if (c  != null) strs.add("NOT_" + c);
					}
				}
				if (nextnext != null) {
					String cs = getCitationString(cm,nextnext);
					if (cs != null) {
						strs.add("NOT_" + cs);
					}
					else {
						String c = type.convert(nextnext);
						if (c != null) strs.add("NOT_" + c);
					}
				}
			}
			return strs;
		}
	}
	
	private class StructureNGramFeature extends StringSetFeature<T> {
		private final int n;
		protected StructureNGramFeature(final String name,
				final int n) {
			super(name);
			this.n = n;
		}
		@Override
		public Set<String> compute(final T span) {
			final List<Span> context = span.getContext();
			final Set<String> grams = new TreeSet<String>();
			for (Span sp: context) {
				List<String> sems = getSemanticString(span,sp);
				Set<String> ngrams = new TreeSet<>();
				stringNgrams(sems,n,ngrams);
				grams.addAll(ngrams);
			}
			return grams;
		}
	}
	
	public class StructureDirectionFeature extends StringSetFeature<T> {
		protected StructureDirectionFeature() {
			super("StructureDirection");
		}
		@Override
		public Set<String> compute(final T span) {
			List<String> types = Arrays.asList("CITINGWORK","[TC]","[OC]","CONTRAST");
			final List<Span> context = span.getContext();
			final Set<String> grams = new TreeSet<String>();
			for (Span sp: context) {
				List<String> sems = getSemanticString(span,sp);
				if (sems.contains("CONTRAST") == false) continue;
				List<Integer> contrastInds = new ArrayList<>();
				List<String>subsems = new ArrayList<>();
				for (int i=0; i < sems.size(); i++) {
					String sem = sems.get(i);
					if (types.contains(sem)) subsems.add(sem);
					if (sem.equals("CONTRAST")) contrastInds.add(i);
				}
				for (int i=0;  i < subsems.size(); i++) {
					String s = subsems.get(i);
					if (s.equals("CONTRAST")) continue;
					if (i < contrastInds.get(0)) grams.add(s + "_CONTRAST_DIR");
					else if (i > contrastInds.get(0) && i < contrastInds.get(contrastInds.size()-1)) {
						grams.add("CONTRAST_" + s + "_DIR"); grams.add(s + "_CONTRAST_DIR");
					} 
					else if (i > contrastInds.get(contrastInds.size()-1)) grams.add("CONTRAST_" + s + "_DIR");
				}
			}
			return grams;
		}
	}
	
	public class ContextUnigramFeature extends ContextNGramFeature {
		public ContextUnigramFeature(final Type type) {
			super("ContextUnigram(" + type.toString().toLowerCase() + ")", 1, type);
		}
	}

	public class ContextBigramFeature extends ContextNGramFeature {
		public ContextBigramFeature(final Type type) {
			super("ContextBigram(" + type.toString().toLowerCase() + ")", 2, type);
		}
	}

	public class ContextTrigramFeature extends ContextNGramFeature {
		public ContextTrigramFeature(final Type type) {
			super("ContextTrigram(" + type.toString().toLowerCase() + ")", 3, type);
		}
	}

	public class StructureUnigramFeature extends StructureNGramFeature {
		public StructureUnigramFeature() {
			super("StructureUnigram", 1);
		}
	}

	public class StructureBigramFeature extends StructureNGramFeature {
		public StructureBigramFeature() {
			super("StructureBigram", 2);
		}
	}

	public class StructureTrigramFeature extends StructureNGramFeature {
		public StructureTrigramFeature() {
			super("StructureTrigram", 3);
		}
	}
	
	  public class NegationCountFeature extends DoubleFeature<T> {
			public NegationCountFeature() {
			  super("NegationCount");
			}

			@Override
			public Double compute(final CitationMention cm) {
				List<SurfaceElement> negs =  getNegationClues(cm);
				double cnt = 0;
				if (negs != null) cnt = Double.valueOf(negs.size());
				return cnt;
			}
		  }

	public class PosSentimentFeature extends StringFeature<T> {
		protected PosSentimentFeature() {
			super("PosSentiment");
		}
		@Override
		public String compute(final T cm) {
			List<SurfaceElement> posSpans = findDictionaryPhrases(cm,"POS");
			List<SurfaceElement> negSpans = findDictionaryPhrases(cm,"NEG");
			if (posSpans.size() > 0){
				for (SurfaceElement pos: posSpans){
					if (inNegationScope(cm.getDocument(),pos) == false) 
						return "TRUE";
				}
			}
			if (negSpans.size() >0) {
				for (SurfaceElement neg: negSpans){
					if (inNegationScope(cm.getDocument(),neg))
						return "TRUE";
				}
			}
			return "FALSE";
		}
	}
	
	public class NegSentimentFeature extends StringFeature<T> {
		protected NegSentimentFeature() {
			super("NegSentiment");
		}
		@Override
		public String compute(final T cm) {
			List<SurfaceElement> posSpans = findDictionaryPhrases(cm,"POS");
			List<SurfaceElement> negSpans = findDictionaryPhrases(cm,"NEG");
			if (negSpans.size() > 0){
				for (SurfaceElement neg: negSpans){
					if (inNegationScope(cm.getDocument(),neg) == false) 
						return "TRUE";
				}
			}
			if (posSpans.size() >0) {
				for (SurfaceElement pos: posSpans){
					if (inNegationScope(cm.getDocument(),pos)) 
						return "TRUE";
				}
			}
			return "FALSE";
		}
	}
	
	
	public class AnySentimentFeature extends StringFeature<T> {
		protected AnySentimentFeature() {
			super("AnySentiment");
		}
		@Override
		public String compute(final T cm) {
			List<SurfaceElement> posSpans = findDictionaryPhrases(cm,"POS");
			List<SurfaceElement> negSpans = findDictionaryPhrases(cm,"NEG");

			if (negSpans.size() > 0 || posSpans.size() > 0){
				return "TRUE";
			}
			return "FALSE";
		}
	}
	
	
	public class RuleBasedOutputFeature extends StringFeature<T> {
		Properties props = null;
		protected RuleBasedOutputFeature(Properties props) {
			super("RuleBasedOutput");
			this.props = props;
		}
		@Override
		public String  compute(final T cm) {
			// this assumes the document with the citation mention has been preprocessed
			RuleBasedSentiment ruleBased = RuleBasedSentiment.getInstance(props);
			String result = ruleBased.processMention(cm);
			String[] scoreStrs = result.split("[\\|]");
			String predict = scoreStrs[3];
			return  predict;
		}
	}
	
	public class RuleBasedOutputFromFileFeature extends StringFeature<T> {
		Map<String,String> results;
		protected RuleBasedOutputFromFileFeature(Map<String,String> res) {
			super("RuleBasedOutput");
			this.results = res;
		}
		@Override
		public String  compute(final T cm) {
			String id = cm.getDocument().getId() + "_" + cm.getId();
			String ruleOut =results.get(id);
/*			if (ruleOut == null) {
				ruleOut = cm.getMetaData("goldSentiment");
			}*/
			return  ruleOut;
		}
	}
	
	public class ContextDependenciesFeature extends StringSetFeature<T> {
		private final Type type;
		protected ContextDependenciesFeature(
				final Type type) {
			super("ContextDependencies");
			this.type = type;
		}
		@Override
		public Set<String> compute(final T span) {
			final List<Span> context = span.getContext();
			List<SynDependency> deps = getContextDependencies(span.getDocument(),context);
			final Set<String> depStrs = new TreeSet<String>();
			for (SynDependency d: deps) {
				  SurfaceElement gov =d.getGovernor();
				  SurfaceElement dep =d.getDependent();
				  String govstr = "";
					if (isCitationMention(gov)) {
						govstr = getCitationString(span,gov);
					} else {
						govstr = type.convert(gov);
					}	
					String depstr = "";
					if (isCitationMention(dep)) {
						depstr = getCitationString(span,dep);
					} else {
						depstr = type.convert(dep);
					}	
					depStrs.add(d.getType() + "_" + govstr + "_" + depstr);
			}
			return depStrs;
		}
	}
		
	private List<SurfaceElement> findDictionaryPhrases(CitationMention cm, String type) {
		List<Span> context = cm.getContext();
		Document doc = cm.getDocument();
		List<SurfaceElement> surfs = new ArrayList<>();
		LinkedHashSet<SemanticItem> phraseSems  = Document.getSemanticItemsByClassTypeSpan(doc, Entity.class, Arrays.asList(type),new SpanList(context), false);
		for (SemanticItem sem: phraseSems) {
			Entity ent = (Entity)sem;
			 String text =  ent.getConcepts().iterator().next().getName();
			if (usedTerms == null || usedTerms.containsKey(text))
				surfs.add(ent.getSurfaceElement());
		}
		return surfs;
	}
	
	private List<SurfaceElement> getNegationClues(CitationMention cm) {
		List<SurfaceElement> negs = findDictionaryPhrases(cm,"NEGATION");
		return negs;
	}
	
	private boolean inNegationScope(Document doc,SurfaceElement surf) {
		Sentence s = surf.getSentence();
		int ind = s.getSurfaceElements().indexOf(surf);
		if (ind < 1) return false;
		SurfaceElement prev = s.getSurfaceElements().get(ind-1);
		if (isNegation(prev)) return true;
		if (ind >1) {
			SurfaceElement prevprev = s.getSurfaceElements().get(ind-2);
			if (isNegation(prevprev)) return true;
		}
		return false;
	}
	
	private boolean isNegation(SurfaceElement surf) {
		LinkedHashSet<SemanticItem> cits = surf.filterSemanticsByClass(Term.class);
		if (cits == null || cits.size() == 0) return false;
		for (SemanticItem s: cits) {
			if (s.getType().equals("NEGATION")) return true;
		}
		return false;
	}
	
	private List<String> getSemanticString(CitationMention cm, Span context) {
		List<String> sems = new ArrayList<>();
		List<SurfaceElement> surfs = cm.getDocument().getSurfaceElementsInSpan(context);
		for (SurfaceElement surf: surfs) {
			LinkedHashSet<SemanticItem> es= surf.filterByEntities();
			LinkedHashSet<SemanticItem> nes = new LinkedHashSet<>();
			if (usedTerms != null) {
				for (SemanticItem se: es) {
					Entity ee = (Entity)se;
					 String text =  ee.getConcepts().iterator().next().getName();
					 if (usedTerms.containsKey(text)) {
						 nes.add(se);
					 }
				}
			} else 
				nes.addAll(es);
			String s = getCitationString(cm,surf);
			if (s != null) {
				sems.add(s);
				continue;
			} 
			if (nes == null || nes.size() ==0) continue;
//			if (isUMLSConcept(surf)) continue;
			Entity ent = (Entity)nes.iterator().next();
			sems.add(ent.getType());
			
		}
		return sems;
	}
	
	private boolean isCitationMention(SurfaceElement surf) {
		LinkedHashSet<SemanticItem> cits = surf.filterSemanticsByClass(CitationMention.class);
		return (cits != null && cits.size() > 0);
	}
	
	private String getCitationString(CitationMention cm, SurfaceElement token) {
		if (cm.getSurfaceElement().equals(token)) return "[TC]";
		else if (isCitationMention(token)) return "[OC]";
		return null;
	}
	
	  private List<Span> getSubsumedSubjectMatter(CitationMention cm, Document doc) {
		  List<Span> sps = new ArrayList<>();
		  List<Span> context = cm.getContext();
		  if (context == null || context.size() == 0) return sps;
		  List<Span> sms = getSubjectMatterSpans(doc);
		  for (Span sp: context) {
			  for (Span sm: sms) {
				  if (Span.subsume(sp, sm)) {
					  sps.add(sm);
				  }
			  }
		  }
		  return sps;
	  }
	  
	  private List<Span> getSubjectMatterSpans(Document doc) {
		  List<Span> sps = new ArrayList<Span>();
		  LinkedHashSet<SemanticItem> cits = Document.getSemanticItemsByClass(doc, CitationMention.class);
		  for (SemanticItem sem: cits) {
			  CitationMention cit = (CitationMention)sem;
			  List<Span> sm = cit.getSubjectMatter();
			  if (sm == null) continue;
			  sps.addAll(sm);
		  }
		  return sps;
	  }
	  
/*	  private boolean inSubjectMatter(SurfaceElement s, List<Span> spans) {
		  if (spans == null) return false;
		  for (Span sp: spans) {
			  if (Span.subsume(sp,s.getSpan().asSingleSpan()) ) return true;
		  }
		  return false;
	  }
	  
		private boolean otherSubjectMatter(SurfaceElement s, List<Span> sms) {
			if (sms == null) return false;
			for (Span sp: sms) {
				if (Span.subsume(sp, s.getSpan().asSingleSpan())) return true;
			}
			return false;
		}*/

	  private List<SynDependency> getContextDependencies(Document doc,  List<Span> context) {
		  List<SynDependency> spDeps = new ArrayList<>();
		  for (Span sp: context) {
			  List<Sentence> sents = doc.getAllSubsumingSentences(sp);
			  for (Sentence sent: sents) {
				  List<SynDependency> sds = sent.getEmbeddings();
				  for (SynDependency sd: sds) {
					  SurfaceElement gov = sd.getGovernor();
					  SurfaceElement dep = sd.getDependent();
					  if (Span.subsume(sp,gov.getSpan().asSingleSpan()) && Span.subsume(sp, dep.getSpan().asSingleSpan())) {
						  spDeps.add(sd);
					  }
				  }
			  }
		  }
		  return spDeps;
	  }
	
}