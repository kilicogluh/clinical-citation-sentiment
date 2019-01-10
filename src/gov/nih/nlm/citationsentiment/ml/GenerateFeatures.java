package gov.nih.nlm.citationsentiment.ml;

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
import java.util.TreeMap;
import java.util.logging.Logger;

import gov.nih.nlm.citationsentiment.CitationFactory;
import gov.nih.nlm.citationsentiment.CitationMention;
import gov.nih.nlm.citationsentiment.Utils;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.io.XMLReader;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.util.FileUtils;
import gov.nih.nlm.ml.feature.DoubleFeature;
import gov.nih.nlm.ml.feature.Feature;
import gov.nih.nlm.ml.feature.FloatFeature;
import gov.nih.nlm.ml.feature.StringDoubleMapFeature;
import gov.nih.nlm.util.Pair;
import liblinear.FeatureNode;

/**
 * Class to generate features for training and testing of neural nets.
 * 
 * @author Halil Kilicoglu
 *
 */
public class GenerateFeatures {
	private static Logger log = Logger.getLogger(GenerateFeatures.class.getName());	

	public static String TRAIN_DIR;
	public static String TEST_DIR;
	public static String LOG_DIR;
	public static String FEATURE_DIR;

	private static Map<String,Integer> featureCounts = new HashMap<>();

	private static List<Integer> trainOutputs = new ArrayList<Integer>();
	private static Map<Integer,String> featureIDs = new TreeMap<Integer,String>();
	private static Map<String,Integer> rFeatureIDs = new TreeMap<String,Integer>();
	private static Map<Integer,String> outputIDs = new TreeMap<Integer,String>();
	private static Map<String,Integer> rOutputIDs = new TreeMap<String,Integer>();

	private static Map<Integer,Double> maxValues = new TreeMap<Integer,Double>();
	private static Map<Integer,Double> minValues = new TreeMap<Integer,Double>();

	private static Map<String,String> dictionaryItems = new HashMap<>();
	private static Properties properties;
	
	private boolean split = false;
	private Map<Class<? extends SemanticItem>,List<String>> annTypes;
	private XMLReader xmlReader;
	private List<CitationMention> trainingInstances = new ArrayList<>();
	private List<CitationMention> testInstances = new ArrayList<>();
	private Map<String,String> ruleBasedResults = new HashMap<>();
	
	public GenerateFeatures(boolean split) throws IOException {
		properties = FileUtils.loadPropertiesFromFile("citation.properties");
		this.split = split;
		this.annTypes = Utils.getAnnotationTypes();
		this.xmlReader = Utils.getXMLReader();
		String file = "";
		try {
			if (split) file = properties.getProperty("termDictionaryTrain"); 
			else file = properties.getProperty("termDictionary"); 
			dictionaryItems = Utils.loadTermDictionary(file);
		} catch (IOException ioe) {
			log.severe("Unable to load  terms from file " + file + ". The program may not work as expected.");
		}
		
		if (split) {
			TRAIN_DIR = properties.getProperty("sentimentTrainDirectory");
		} else 
			TRAIN_DIR = properties.getProperty("sentimentAllDirectory");

		LOG_DIR = properties.getProperty("logDirectory");
		FEATURE_DIR = properties.getProperty("featureDirectory");

		if (split)  {
			TEST_DIR = properties.getProperty("sentimentTestDirectory");
		} 
	}
	
	public static void main(String[] args) throws Exception {
		boolean split = Boolean.parseBoolean(args[0]);
		GenerateFeatures generator = new GenerateFeatures(split);
		generator.setRuleBasedResults(Utils.loadRuleBasedResultsFromFile(properties.getProperty("ruleBasedResultFile"))); 
		generator.setTrainingInstances(generator.preProcessDir(TRAIN_DIR));
		if (split)  {
			generator.setTestInstances(generator.preProcessDir(TEST_DIR));
		} 
		String featureStr = generator.processInstances();
		writeCsvFeatures(FEATURE_DIR + File.separator +  "all" + (split?  "_split" : "") + ".csv", featureStr);
	}
	
	public List<CitationMention> getTrainingInstances() {
		return trainingInstances;
	}
	
	public void setTrainingInstances(List<CitationMention> instances) {
		this. trainingInstances = instances;
	}
	
	public List<CitationMention> getTestInstances() {
		return testInstances;
	}
	
	public void setTestInstances(List<CitationMention> instances) {
		this. testInstances = instances;
	}
	
	public Map<String, String> getRuleBasedResults() {
		return ruleBasedResults;
	}

	public void setRuleBasedResults(Map<String, String> ruleBasedResults) {
		this.ruleBasedResults = ruleBasedResults;
	}

	public List<CitationMention> preProcessDir(String dir) throws Exception {
		List<CitationMention> instances = new ArrayList<>();
		List<String> files = FileUtils.listFiles(dir, false, "xml");
		Collections.sort(files);
		int fileNum = 0;
		for (String filename: files) {
			String filenameNoExt = filename.replace(".xml", "");
			filenameNoExt = filenameNoExt.substring(filenameNoExt.lastIndexOf(File.separator)+1);
			log.info("Processing " + filenameNoExt + ":" + ++fileNum);
			List<CitationMention> fileInstances = preProcessFile(filename);
			instances.addAll(fileInstances);
		}	
		return instances;
	}
	
	public  List<CitationMention> preProcessFile(String filename) throws Exception {  
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
		return preProcessDocument(doc,gold);
	}

	public  List<CitationMention> preProcessDocument(Document doc,Map<String,String> gold) throws Exception {
		Utils.annotateStrings(doc, dictionaryItems, properties);
		Utils.removeSubsumedTerms(doc);
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
			cm.setContext(Utils.getCitationContext(cm,false));
			cm.setMetaData("goldSentiment", goldSentiment);
			citationMentions.add(cm);
			seen.add(surf);
		}
		return citationMentions;
	}
	
		
	public String  processInstances() throws Exception {
		StringBuilder buf = new StringBuilder();
		featureIDs = new TreeMap<Integer,String>();
		rFeatureIDs = new TreeMap<String,Integer>();
		outputIDs = new TreeMap<Integer,String>();
		rOutputIDs = new TreeMap<String,Integer>();

		int t = 0;
		int posC = 0; int negC = 0; int neuC=0;
		List<Feature<CitationMention,?>> features = getNNFeatures();
		List<List<FeatureNode>> instanceFeatures = new ArrayList<>();

		for (final CitationMention instance : trainingInstances) {
			List<FeatureNode> fn = extractFeatures(true,instance, features);
			instanceFeatures.add(fn);
			t++;
			String s = instance.getMetaData("goldSentiment");
			if (s.equals("NEUTRAL")) neuC++;
			else if (s.equals("POSITIVE")) posC++;
			else if (s.equals("NEGATIVE")) negC++;
		}
		log.info("Number of training instances: " + t);
		log.info("POS " + posC +  " NEG " + negC + " NEU " + neuC);
		String trainCsv = convertToCsv(trainingInstances,instanceFeatures);
		buf.append(trainCsv);
		if (LOG_DIR != null) 
			writeLog(trainingInstances,features,LOG_DIR + File.separator +  "train" + (split ? "_split" : "") + ".log");

		t=0;
		posC = 0;  negC = 0; neuC=0;
		instanceFeatures = new ArrayList<>();
		for (final CitationMention instance : testInstances) {
			List<FeatureNode> fn = extractFeatures(false,instance, features);
			instanceFeatures.add(fn);
			t++;
			String s = instance.getMetaData("goldSentiment");
			if (s.equals("NEUTRAL")) neuC++;
			else if (s.equals("POSITIVE")) posC++;
			else if (s.equals("NEGATIVE")) negC++;
		}
		log.info("Number of test instances: " + t);
		log.info("POS " + posC +  " NEG " + negC + " NEU " + neuC);
		String testCsv = convertToCsv(testInstances,instanceFeatures);
		buf.append(testCsv);
		if (split && LOG_DIR != null) 
			writeLog(testInstances,features,LOG_DIR + File.separator +  "test_split.log");
		return buf.toString();
	}
	
	private static void writeCsvFeatures(String filename, String csv) throws IOException {
		PrintWriter pw = new PrintWriter(filename);
		pw.write(csv);
		pw.write("\n");
		pw.flush();
		pw.close();
	}
	
	private String convertToCsv(List<CitationMention> mentions, List<List<FeatureNode>> instanceFeatures)  {
		StringBuffer buf = new StringBuffer();
		for (Integer id: featureIDs.keySet()) {
//			if (filter(id)) continue;
			buf.append(",f" + (id-1));
		}
		buf.append("\n");
		for (int i=0; i < instanceFeatures.size(); i++) {
			log.fine("Instance feature size: " + i);
			//			  buf.append(outputIDs.get(trainOutputs.get(i)));
			buf.append(mentions.get(i).getMetaData("goldSentiment"));
			List<FeatureNode> fns=instanceFeatures.get(i);
			List<Integer> nonZero = new ArrayList<>();
			List<Double> nonZeroValues = new ArrayList<>();

			for (FeatureNode fna: fns) {
				nonZero.add(fna.index-1);
				nonZeroValues.add(fna.value);
			}
			for (int j=0; j < featureIDs.size(); j++) {
//				if (filter(j+1)) continue;
				if (nonZero.contains(j)) {
					int ind = nonZero.indexOf(j);
					buf.append("," + nonZeroValues.get(ind));
				} else {
					buf.append(",0."); 
				}
			}
			buf.append("\n");
		}
		return buf.toString();
	}

	private static void writeLog(Iterable<CitationMention> instances, List<Feature<CitationMention,?>> features, String filename) throws Exception {
		if (filename == null) return;
		final StringBuilder logBuilder = new StringBuilder();
		int num =0;
		for (Integer id: featureIDs.keySet()) {
			//		  pw.write("@attribute FEATURE_" + id + "_" + featureIDs.get(id) + " numeric");
			logBuilder.append("FEATURE_" + (id-1) + " " + featureIDs.get(id) + " numeric");
			logBuilder.append("\n");
		}
		for (CitationMention sample: instances) {
			logBuilder.append("---------- Sample " + ++num + " ----------\n");
			logBuilder.append(sample.toString() + "\n");
			for (final Feature<CitationMention,?> feature : features) {	  
				logBuilder.append(feature.getName());
				logBuilder.append("=");
				final Object value = feature.get(sample);
				//   		Collection<String> allValues = feature.getAllValues(sample);
				if (value == null) {
					logBuilder.append("NULL");
				}
				else {
					final String str = value.toString();
					assert str.contains("\n") == false;
					logBuilder.append(str);
					logBuilder.append("\n");
					/*    			if (feature.isContinuous()) {
    				int id = rFeatureIDs.get(feature.getName());
    			} else {
    				for (String val: allValues) {
					  int id = rFeatureIDs.get(feature.getName()+"="+val);
    				}
				  }*/
				}
			}
		}
		PrintWriter logPw = new PrintWriter(filename);
		logPw.write(logBuilder.toString());
		logPw.flush();
		logPw.close();
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

	public List<FeatureNode> extractFeatures(boolean train, final CitationMention sample, List<Feature<CitationMention,?>> features) {
		final List<FeatureNode> featureNodes = new ArrayList<FeatureNode>();
		for (final Feature<CitationMention,?> feature : features) {
			final String featureName = feature.getName();
			if (feature.isContinuous() && feature.isMulti()) {
				assert feature instanceof StringDoubleMapFeature;
				final StringDoubleMapFeature<CitationMention> sdmFeature =
						(StringDoubleMapFeature<CitationMention>) feature;
				for (final Pair<String,Double> pair : sdmFeature.getValues(sample)) {
					final String fname = featureName + "::" + pair.getFirst();
					if (registeredFeature(fname) == false && !train) continue;
					final Double value = pair.getSecond();
					if (value == null) {
						continue;
					}
					if (registeredFeature(fname) == false) {
						registerFeature(fname);
						//					  System.out.println("@attribute FEATURE_" + rFeatureIDs.get(featureName) + "_" + fname + " numeric");
						//					  pw.write("@attribute FEATURE_" + rFeatureIDs.get(featureName) + "_" + fname.replaceAll(" ", "_") + " numeric");
						//					  pw.write("\n");
					}
					final int featureNum = rFeatureIDs.get(fname);
					if (maxValues.containsKey(featureNum)) {
						if (value > maxValues.get(featureNum)) maxValues.put(featureNum, value);
					} else maxValues.put(featureNum, value);
					if (minValues.containsKey(featureNum)) {
						if (value < minValues.get(featureNum)) minValues.put(featureNum, value);
					} else minValues.put(featureNum, value);
					featureNodes.add(new FeatureNode(featureNum,value));
					if (train) {
						int cnt = (featureCounts.containsKey(fname) ? featureCounts.get(fname) : 0);
						featureCounts.put(fname, ++cnt);
					}
				}
			}
			else if (feature.isContinuous()) {
				assert feature instanceof FloatFeature ||
				feature instanceof DoubleFeature;
				if (registeredFeature(featureName) == false && !train) continue;
				if (registeredFeature(featureName) == false) {
					registerFeature(featureName);
					//				  System.out.println("@attribute FEATURE_" + rFeatureIDs.get(featureName) + "_" + featureName + " numeric");   
					//				  pw.write("@attribute FEATURE_" + rFeatureIDs.get(featureName) + "_" + featureName.replaceAll(" ", "_") + " numeric");   
					//				  pw.write("\n");
				}
				final int featureNum = rFeatureIDs.get(featureName);
				final Double value = feature.getDouble(sample);
				if (value != null) {
					if (maxValues.containsKey(featureNum)) {
						if (value > maxValues.get(featureNum)) maxValues.put(featureNum, value);
					} else maxValues.put(featureNum, value);
					if (minValues.containsKey(featureNum)) {
						if (value < minValues.get(featureNum)) minValues.put(featureNum, value);
					} else minValues.put(featureNum, value);

					featureNodes.add(new FeatureNode(featureNum, value));
					if (train) {
						int cnt = (featureCounts.containsKey(featureName) ? featureCounts.get(featureName) : 0);
						featureCounts.put(featureName, ++cnt);
					}
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
						if (registeredFeature(nvalue) == false && !train) continue;
						if (registeredFeature(nvalue) == false) {
							registerFeature(nvalue);
							//						  System.out.println("@attribute FEATURE_" + rFeatureIDs.get(nvalue) + "_" + nvalue + " string");  
							//						  pw.write("@attribute FEATURE_" + rFeatureIDs.get(nvalue) + "_" + nvalue.replaceAll(" ", "_")  + " string");  
							//						  pw.write("\n");
						}
						final int featureNum = rFeatureIDs.get(nvalue);
						if (values.contains("TRUE"))
							featureNodes.add(new FeatureNode(featureNum, 1.0));
						else if (values.contains("FALSE")) 
							featureNodes.add(new FeatureNode(featureNum, 0.0));
						minValues.put(featureNum, 0.0);
						maxValues.put(featureNum, 1.0);
						if (train) {
							int cnt = (featureCounts.containsKey(nvalue) ? featureCounts.get(nvalue) : 0);
							featureCounts.put(nvalue, ++cnt);
						}
						continue;
					}
				}
				for (final String value : values) {
					final String nvalue = featureName + "=" + value;
					if (registeredFeature(nvalue) == false && !train) continue;
					if (registeredFeature(nvalue) == false) {
						registerFeature(nvalue);
						//					  System.out.println("@attribute FEATURE_" + rFeatureIDs.get(nvalue) + "_" + nvalue + " string");  
						//					  pw.write("@attribute FEATURE_" + rFeatureIDs.get(nvalue) + "_" + nvalue.replaceAll(" ", "_")  + " string");  
						//					  pw.write("\n");
					}
					final int featureNum = rFeatureIDs.get(nvalue);
					featureNodes.add(new FeatureNode(featureNum, 1.0));
					minValues.put(featureNum, 0.0);
					maxValues.put(featureNum, 1.0);
					if (train) {
						int cnt = (featureCounts.containsKey(nvalue) ? featureCounts.get(nvalue) : 0);
						featureCounts.put(nvalue, ++cnt);
					}
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
					log.warning(" " + feature.getName());
				}
			}
		}
		Collections.sort(featureNodes, Features.FEATURE_NODE_COMPARATOR);
		final String output = sample.getMetaData("goldSentiment");
		if (rOutputIDs.containsKey(output) == false) {
			final int outputID = outputIDs.size() + 1;
			outputIDs.put(outputID, output);
			rOutputIDs.put(output, outputID);
		}
		final int outputID = rOutputIDs.get(output);
		trainOutputs.add(outputID);
		log.fine("Feature node size:" + featureNodes.size());
		return featureNodes;
	}

	private static boolean registeredFeature(final String featureName) {
		return rFeatureIDs.containsKey(featureName);
	}
	
	private static void registerFeature(final String featureName) {
		final int featureNum = featureIDs.size() + 1;
		featureIDs.put(featureNum, featureName);
		rFeatureIDs.put(featureName, featureNum);
	}
}
