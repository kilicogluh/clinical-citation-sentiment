package gov.nih.nlm.citationsentiment;

import java.util.Map;
import java.util.logging.Logger;

import gov.nih.nlm.ling.brat.TermAnnotation;
import gov.nih.nlm.ling.core.Document;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.sem.Entity;
import gov.nih.nlm.ling.sem.SemanticItem;
import gov.nih.nlm.ling.sem.SemanticItemFactory;

/**
 * Class to create citation-related semantic objects.
 * 
 * @author Halil Kilicoglu
 *
 */
public class CitationFactory extends SemanticItemFactory {
	private static Logger log = Logger.getLogger(CitationFactory.class.getName());	

	public CitationFactory(Document doc, Map<Class<? extends SemanticItem>,Integer> counterMap) {
		super(doc,counterMap);
	}
	
	
	/**
	 * Creates a new {@link CitationMention} object from an existing standoff {@link TermAnnotation} object.
	 * 
	 * @param doc	the document that the expression is associated with
	 * @param t		the corresponding <code>TermAnnotation</code> standoff annotation
	 * @return 		a <code>CitationMention</code> object
	 */
	public CitationMention newCitationMention(Document doc, TermAnnotation t) {
		CitationMention cit = newCitationMention(doc,t.getType(),t.getSpan(),t.getSpan(),t.getText());
		cit.setId(t.getId());
		return cit;
	}
	
	/**
	 * Creates a new <code>CitationMention</code> object from type and span information.
	 * 
	 * @param doc		the document that the coreferential expression is associated with
	 * @param type		the specific type of the coreferential expression
	 * @param sp		the span of the term
	 * @param headSp	the head span of term
	 * @param text  	the text of the term
	 * @return   		a <code>CitationMention</code> object
	 */
	public CitationMention newCitationMention(Document doc, String type, SpanList sp, SpanList headSp, String text) {	
		SurfaceElement si = doc.getSurfaceElementFactory().createSurfaceElementIfNecessary(doc, sp, headSp, true);
		CitationMention men = new CitationMention("C" + getNextId(CitationMention.class),type,sp,headSp,si);
		men.setSurfaceElement(si);
		si.addSemantics(men);
		doc.addSemanticItem(men);
		return men;
	}
	
/*	public CitationMention newCitationMention(Document doc, String type, SpanList sp, SpanList headSp, String text, Citation citation) {	
		CitationMention men = newCitationMention(doc,type,sp,headSp,text);
		men.setCitation(citation);
		return men;
	}*/
	
	public CitationMention newCitationMention(Document doc, Entity entity, String text) {	
		return newCitationMention(doc,entity.getType(),entity.getSpan(),entity.getHeadSpan(),text);
	}
	
	/**
	 * Creates a new <code>CitationMention</code> object from type and textual unit information.<p>
	 * If the type specified for the mention is unrecognized, it will attempt to identify the
	 * fine-grained type.
	 * 
	 * @param doc	the document that the mention is associated with
	 * @param type  the specific type of the coref. mention
	 * @param se  	the corresponding textual unit
	 * @return  	a <code>CitationMention</code> object
	 */
	public CitationMention newCitationMention(Document doc, String type, SurfaceElement se) {	
		return newCitationMention(doc,type,se);
	}
}
