package gov.nih.nlm.citationsentiment.ml;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

import gov.nih.nlm.citationsentiment.CitationMention;
import gov.nih.nlm.citationsentiment.RuleBasedSentiment;
import gov.nih.nlm.citationsentiment.Utils;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.core.Word;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.util.FileUtils;
import gov.nih.nlm.ml.feature.DoubleFeature;
import gov.nih.nlm.ml.feature.Feature;
import gov.nih.nlm.ml.feature.FloatFeature;
import gov.nih.nlm.ml.feature.StringDoubleMapFeature;
import gov.nih.nlm.util.Pair;
import liblinear.FeatureNode;

/**
 * Class for generating features for prediction. Features are extracted as a comma-separated string.  
 * 
 * @author Halil Kilicoglu
 *
 */
public class GenerateFeaturesForPrediction {
	private static Logger log = Logger.getLogger(GenerateFeaturesForPrediction.class.getName());	

	private static Map<Integer,String> featureIDs = new TreeMap<Integer,String>();
	private static Map<String,Integer> rFeatureIDs = new TreeMap<String,Integer>();

	private static Properties properties;	
	private static Map<String,String> dictionaryItems = new HashMap<>();
	private static RuleBasedSentiment ruleBasedMethod;
	
	private Map<String,String> ruleBasedResults = new LinkedHashMap<>();

	/**
	 * Loads properties, feature list and dictionary terms, and creates an instance of the 
	 * rule-based method. 
	 */
	public GenerateFeaturesForPrediction() {
		try {
			properties = FileUtils.loadPropertiesFromFile("citation.properties");
		} catch (IOException ioe) {
			log.severe("Unable to load properties file. Exiting..");
			System.exit(1);
		}
		String file = "";
		try {
			file = properties.getProperty("featureFile"); 
			readFeatures(file);
		} catch (IOException ioe) {
			log.severe("Features cannot be loaded from " + file + ". Exiting...");
			System.exit(1);
		}
		try {
			file = properties.getProperty("termDictionary");
			dictionaryItems = Utils.loadTermDictionary(file);
		} catch (IOException ioe) {
			log.severe("Unable to load  terms from file " + file + ". The program may not work as expected.");
		}
		ruleBasedMethod = RuleBasedSentiment.getInstance(properties);
	}
	
	public static void main(String[] args) throws Exception {
		GenerateFeaturesForPrediction generator = new GenerateFeaturesForPrediction();
		String filename = args[0];
		String text = FileUtils.stringFromFile(filename, "UTF-8");
		String out = generator.process(filename,text);
		
		PrintWriter pw = new PrintWriter(args[1]);
		pw.println(out);
		pw.flush(); pw.close();
	}
	
	private static String instanceData(CitationMention cm) {
	  	List<Span> contexts  = cm.getContext();
	  	Document doc = cm.getDocument();
	  	StringBuffer buf = new StringBuffer();
	  	buf.append(cm.getId() + ":" + cm.getText() + "|");
	  	for (Span context: contexts) {
	  		List<SurfaceElement> surfs = doc.getSurfaceElementsInSpan(context);
			StringBuffer wbuf = new StringBuffer();
				for (SurfaceElement surf: surfs) {
					if (Utils.isCitationMention(surf)) {
						String cs = Utils.getCitationString(cm,surf);
						if (cs != null) wbuf.append(cs);
					} else {
						for (Word w: surf.toWordList()) {
//						if (Pattern.matches("\\p{Punct}", w.getText())) continue;
							wbuf.append(w.getText() + " ");
						}
					}			
				}
				buf.append(wbuf.toString().trim() + "|");
				wbuf = new StringBuffer();
				for (SurfaceElement surf: surfs) {
					if (Utils.isCitationMention(surf)) {
						String cs = Utils.getCitationString(cm,surf);
						if (cs != null) wbuf.append(cs);
					} else {

						for (Word w: surf.toWordList()) {
								wbuf.append(w.getPos() + " ");
							}
					}			
				}
				buf.append(wbuf.toString().trim() +"|" );
	  	}
	  	return buf.toString().trim();
}
	
	private void readFeatures(String filename) throws IOException  {
		List<String> lines = FileUtils.linesFromFile(filename, "UTF-8");
		for (String l: lines) {
			if (l.startsWith("FEATURE_") == false) continue;
			String[] els = l.split(" ");
			Integer id = Integer.parseInt(els[0].substring(els[0].indexOf("_")+1, els[0].length()));
			String name = els[1];
			featureIDs.put(id,name);
			rFeatureIDs.put(name,id);
		}
	}
	
	/**
	 * The main method. Takes in a string of a document with a given id and generates a feature representation.
	 * The string is expected to be one sentence per line, and each citation mention should be in the format:
	 *  <pre>{@code <cit id="C1">[1-5]</cit>}</pre> 
	 * 
	 * @param id	The id to associate with the string
	 * @param text	The input string
	 * @return	The comma-separated feature representation
	 */
	public String process(String id, String text)  {
		Document doc =preProcessString(id,text);
		return processInstances(doc);
	}
	
	private Document preProcessString(String id, String input)  {  
		Document doc =  ruleBasedMethod.preProcessString(id, input, properties);
		getRuleBasedPredictions(doc);
		reannotate(doc);
		return doc;
	}
	
	private void reannotate(Document doc) {
		LinkedHashSet<SemanticItem> sems = doc.getAllSemanticItems();
		for (SemanticItem sem: sems) {
			if (sem instanceof CitationMention == false)
				doc.removeSemanticItem(sem);
		}
		Utils.annotateStrings(doc, dictionaryItems, properties);
		Utils.removeSubsumedTerms(doc);
	}
	
	private void  getRuleBasedPredictions(Document doc) {
		String str =  ruleBasedMethod.processMentions(doc);
		String[] strs = str.split("\\n");
		ruleBasedResults = new LinkedHashMap<>();
		for (int i=0; i < strs.length; i++) {
			String r = strs[i];
			String[] rs = r.split("\\|");
			ruleBasedResults.put(rs[0] + "_" + rs[1], rs[3]);
		}
	}
	
	private String  processInstances(Document doc)  {
		List<CitationMention> instances = new ArrayList<>();
		LinkedHashSet<SemanticItem> sems = Document.getSemanticItemsByClass(doc, CitationMention.class);
		for (SemanticItem sem: sems) {
			CitationMention s = (CitationMention)sem;
			instances.add(s);
		}
	
		StringBuilder buf = new StringBuilder();
		int t = 0;
		List<Feature<CitationMention,?>> features = getNNFeatures();
		List<List<FeatureNode>> instanceFeatures = new ArrayList<>();

		for (final CitationMention instance : instances) {
			List<FeatureNode> fn = extractFeatures(instance, features);
			instanceFeatures.add(fn);
			t++;
		}
		log.info("Number of test instances: " + t);
		String testCsv = convertToCsv(instances,instanceFeatures);
		buf.append(testCsv);
		writeLog(instances,features,"log" + File.separator +  "predict.log");
		return buf.toString();
	}
	
	private String convertToCsv(List<CitationMention> mentions, List<List<FeatureNode>> instanceFeatures)  {
		StringBuffer buf = new StringBuffer();
		log.fine("Number of instances: " + instanceFeatures.size());
		for (int i=0; i < instanceFeatures.size(); i++) {
			CitationMention m = mentions.get(i);
			String instanceStr = instanceData(m);
			buf.append(instanceStr);
			List<FeatureNode> fns=instanceFeatures.get(i);
			List<Integer> nonZero = new ArrayList<>();
			List<Double> nonZeroValues = new ArrayList<>();

			for (FeatureNode fna: fns) {
				nonZero.add(fna.index);
				nonZeroValues.add(fna.value);
			}
			for (int j=0; j < featureIDs.size(); j++) {
				buf.append( (j == 0 ? "": ","));
				if (nonZero.contains(j)) {
					int ind = nonZero.indexOf(j);
					buf.append(nonZeroValues.get(ind));
				} else {
					buf.append("0."); 
				}
			}
			buf.append("\n");
		}
		return buf.toString();
	}

	private static void writeLog(Iterable<CitationMention> instances, List<Feature<CitationMention,?>> features, String filename)  {
		if (filename == null) return;
		final StringBuilder logBuilder = new StringBuilder();
		int num =0;
/*		for (Integer id: featureIDs.keySet()) {
			logBuilder.append("FEATURE_" +id + " " + featureIDs.get(id) + " numeric");
			logBuilder.append("\n");
		}*/
		for (CitationMention sample: instances) {
			logBuilder.append("---------- Sample " + ++num + " ----------\n");
			logBuilder.append(sample.toString() + "\n");
			for (final Feature<CitationMention,?> feature : features) {	  
				logBuilder.append(feature.getName());
				logBuilder.append("=");
				final Object value = feature.get(sample);
//				Collection<String> allValues = feature.getAllValues(sample);
				if (value == null) {
					logBuilder.append("NULL");
				}
				else {
					final String str = value.toString();
					assert str.contains("\n") == false;
					logBuilder.append(str);
					logBuilder.append("\n");
/*					if (feature.isContinuous()) {
						int id = rFeatureIDs.get(feature.getName());
					} else {
	    				for (String val: allValues) {
	    					int id = rFeatureIDs.get(feature.getName()+"="+val);
	    				}
					}*/
				}
			}
		}
		try {
			PrintWriter logPw = new PrintWriter(filename);
			logPw.write(logBuilder.toString());
			logPw.flush();
			logPw.close();
		} catch (IOException ioe) {
			log.severe("Unable to write to log file " + filename);
			ioe.printStackTrace();
		}
	}


	private List<Feature<CitationMention,?>> getNNFeatures() {
		List<Feature<CitationMention,?>> features = new ArrayList<>();
		features = new ArrayList<Feature<CitationMention,?>>();
		final Features fs = new Features();

		features.add((Feature<CitationMention,?>)fs.new ContextUnigramFeature(Features.Type.CAT));
		features.add((Feature<CitationMention,?>)fs.new ContextBigramFeature(Features.Type.CAT));
		features.add((Feature<CitationMention,?>)fs.new ContextTrigramFeature(Features.Type.CAT));
		features.add((Feature<CitationMention,?>)fs.new NegationCountFeature());
		features.add((Feature<CitationMention,?>)fs.new NegationNGramFeature("NegationNGramUnigram(stemmed)", Features.Type.STEMMED));

		// Sentiment features
		features.add((Feature<CitationMention,?>)fs.new PosSentimentFeature());
		features.add((Feature<CitationMention,?>)fs.new NegSentimentFeature());
		features.add((Feature<CitationMention,?>)fs.new AnySentimentFeature());

		//Structure features
		features.add((Feature<CitationMention,?>)fs.new StructureUnigramFeature());
		features.add((Feature<CitationMention,?>)fs.new StructureBigramFeature());
		features.add((Feature<CitationMention,?>)fs.new StructureTrigramFeature());
		features.add((Feature<CitationMention,?>)fs.new StructureDirectionFeature());

		features.add((Feature<CitationMention,?>)fs.new RuleBasedOutputFromFileFeature(ruleBasedResults));
		log.info("Features: " + features);
		return features;
	}

	public List<FeatureNode> extractFeatures(final CitationMention sample, List<Feature<CitationMention,?>> features) {
		final List<FeatureNode> featureNodes = new ArrayList<FeatureNode>();
		for (final Feature<CitationMention,?> feature : features) {
			final String featureName = feature.getName();
			if (feature.isContinuous() && feature.isMulti()) {
				assert feature instanceof StringDoubleMapFeature;
				final StringDoubleMapFeature<CitationMention> sdmFeature =
						(StringDoubleMapFeature<CitationMention>) feature;
				for (final Pair<String,Double> pair : sdmFeature.getValues(sample)) {
					final String fname = featureName + "::" + pair.getFirst();
					if (registeredFeature(fname) == false) continue;
					final Double value = pair.getSecond();
					if (value == null) {
						continue;
					}
					final int featureNum = rFeatureIDs.get(fname);
					featureNodes.add(new FeatureNode(featureNum,value));
				}
			}
			else if (feature.isContinuous()) {
				assert feature instanceof FloatFeature ||
				feature instanceof DoubleFeature;
				if (registeredFeature(featureName) == false) continue;
				final int featureNum = rFeatureIDs.get(featureName);
				final Double value = feature.getDouble(sample);
				if (value != null) {
					featureNodes.add(new FeatureNode(featureNum, value));
				}
			}
			else {
				final List<String> values;
				if (feature.isMulti()) {
					values = new ArrayList<String>();
					values.addAll(feature.getAllValues(sample));
					assert values.isEmpty() || values.get(0) != null;
				}
				else {
					values = Collections.singletonList(feature.getString(sample));
					if (values.contains("TRUE") || values.contains("FALSE")) {
						final String nvalue = featureName;
						if (registeredFeature(nvalue) == false) continue;
						final int featureNum = rFeatureIDs.get(nvalue);
						if (values.contains("TRUE"))
							featureNodes.add(new FeatureNode(featureNum, 1.0));
						else if (values.contains("FALSE")) 
							featureNodes.add(new FeatureNode(featureNum, 0.0));
						continue;
					}
				}
				for (final String value : values) {
					final String nvalue = featureName + "=" + value;
					if (registeredFeature(nvalue) == false) continue;
					final int featureNum = rFeatureIDs.get(nvalue);
					featureNodes.add(new FeatureNode(featureNum, 1.0));
				}
			}
		}
		if (featureNodes.isEmpty()) {
			log.warning("No features for sample");
			log.finest("Sample: " + sample);
		}
		/*    for (FeatureNode fn: featureNodes) {
System.out.println("FEature:" + fn.index + "|" + featureIDs.get(fn.index) + "|" +  fn.value);
fn.value = scale(fn,0.0,1.0);
System.out.println("FEature:" + fn.index + "|" + featureIDs.get(fn.index) + "|" +  fn.value);
}*/

		// Error checking
		final Set<Integer> indexes = new HashSet<Integer>();
		for (final FeatureNode featureNode : featureNodes) {
			if (indexes.add(featureNode.index) == false) {
				log.severe("Found duplicate index!  Features:");
				for (final Feature<CitationMention,?> feature : features) {
					log.warning("  " + feature.getName());
				}
			}
		}
		Collections.sort(featureNodes, Features.FEATURE_NODE_COMPARATOR);
		log.fine("Feature node size:" + featureNodes.size());
		return featureNodes;
	}

	private static boolean registeredFeature(final String featureName) {
		return rFeatureIDs.containsKey(featureName);
	}
	
}
