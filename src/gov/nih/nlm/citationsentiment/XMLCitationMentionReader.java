package gov.nih.nlm.citationsentiment;

import java.util.logging.Level;
import java.util.logging.Logger;

import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.Sentence;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.io.XMLEntityReader;
import gov.nih.nlm.ling.sem.SemanticItem;
import nu.xom.Element;

/**
 * Class to read citation mention information from XML.
 * 
 * @author Halil Kilicoglu
 *
 */
public class XMLCitationMentionReader extends XMLEntityReader {
	private static final Logger log = Logger.getLogger(XMLCitationMentionReader.class.getName());
	
	public XMLCitationMentionReader() {}

	@Override
	public SemanticItem read(Element element, Document doc) {
		String id = element.getAttributeValue("id");
		String spStr = element.getAttributeValue("charOffset");
		String headSpStr = element.getAttributeValue("headOffset");
		String sentiment = element.getAttributeValue("sentiment");
		String refType = element.getAttributeValue("refType");
		String text = element.getAttributeValue("text");
		SpanList sp = new SpanList(spStr);
		SpanList headSp = null;
		if (headSpStr != null) headSp = new SpanList(headSpStr);
		Sentence sent = doc.getSubsumingSentence(sp.getSpans().get(0));
		if (sent == null) {
			log.log(Level.WARNING,"No sentence can be associated with the XML: {0}", new Object[]{element.toXML()});
			return null;
		}
		CitationFactory sif = (CitationFactory)doc.getSemanticItemFactory();
		CitationMention cm = sif.newCitationMention(doc, refType, sp, headSp, text);
		cm.setId(id);
		cm.setSentiment(CitationMention.Sentiment.valueOf(sentiment));
		sent.synchSurfaceElements(cm.getSurfaceElement());
		sent.synchEmbeddings();
	   	log.log(Level.FINEST,"Generated entity {0} with the head {1}. ", 
	   			new String[]{cm.toString(), cm.getSurfaceElement().getHead().toString()});
	   String contextSpan = element.getAttributeValue("context");
	   if (contextSpan != null) {
			   SpanList contextSp = new SpanList(contextSpan);
			   cm.setContext(contextSp.getSpans());
	   }
	   String subjectMatterSpan = element.getAttributeValue("subjectMatter");
	   if (subjectMatterSpan != null) {
			   SpanList smSp = new SpanList(subjectMatterSpan);
			   cm.setSubjectMatter(smSp.getSpans());
	   }
	   	return cm;
	}
}
