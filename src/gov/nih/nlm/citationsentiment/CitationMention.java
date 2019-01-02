package gov.nih.nlm.citationsentiment;

import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import gov.nih.nlm.ling.core.Span;
import gov.nih.nlm.ling.core.SpanList;
import gov.nih.nlm.ling.core.SurfaceElement;
import gov.nih.nlm.ling.sem.AbstractTerm;
import gov.nih.nlm.ling.sem.Ontology;
import gov.nih.nlm.util.attr.Attributes;
import gov.nih.nlm.util.attr.HasAttributes;
import nu.xom.Attribute;
import nu.xom.Element;

/**
 * A representation of a mention of a citation in text. A mention is associated with a context span (
 * in the simplest case, the span of the sentence it is in) and may consist of of several 
 * references (e.g., <i>[1-2]</i>). 
 * 
 * @author Halil Kilicoglu
 *
 */
public class CitationMention extends AbstractTerm implements HasAttributes {
	
	public enum Sentiment {
		POSITIVE, NEGATIVE, NEUTRAL, NONE;
	}
	
//	private Citation citation;
	private Sentiment sentiment;
	private Map<String,String> metaDataMap;
	private Attributes attrs;
	private List<Span> context;
	private List<Span> subjectMatter;
	private List<Span> simplifiedContext;
	
	public CitationMention(String id) {
		super(id);
	}

	public CitationMention(String id, String type, SpanList sp) {
		super(id, type, sp);
		sentiment = Sentiment.NONE;
	}

	public CitationMention(String id, String type, SpanList sp, SurfaceElement se) {
		super(id, type, sp, se);
		sentiment = Sentiment.NONE;
	}

	public CitationMention(String id, String type, SpanList sp, SpanList headSp, SurfaceElement se) {
		super(id, type, sp, headSp, se);
		sentiment = Sentiment.NONE;
	}
	
	public CitationMention(String id, String type, SpanList sp, SpanList headSp, SurfaceElement se, Sentiment sentiment) {
		this(id, type, sp, headSp, se);
		this.sentiment = sentiment;
	}

	@Override
	public Set<String> getSemtypes() {
		return getAllSemtypes();
	}

	@Override
	public LinkedHashSet<String> getAllSemtypes() {
		LinkedHashSet<String> sems = new LinkedHashSet<>();
//		sems.add(citation.getType());
		sems.add("CitationSentiment");
		return sems;
	}

	@Override
	public String toShortString() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	public void setAttrs(Attributes attrs) {
		this.attrs = attrs;
	}

	public void setContext(List<Span> context) {
		this.context = context;
	}

	public void setSubjectMatter(List<Span> subj) {
		this.subjectMatter = subj;
	}
	
	public void setSimplifiedContext(List<Span> context) {
		this.simplifiedContext = context;
	}
	
	public Attributes getAttrs() {
		return attrs;
	}

	public List<Span> getContext() {
		return context;
	}
	
	public List<Span> getSubjectMatter() {
		return subjectMatter;
	}
	
	public List<Span> getSimplifiedContext() {
		return simplifiedContext;
	}

	@Override
	public Element toXml() {
		Element el = new Element("Term");
		el.addAttribute(new Attribute("xml:space", 
		          "http://www.w3.org/XML/1998/namespace", "preserve"));
		el.addAttribute(new Attribute("id",id));
		el.addAttribute(new Attribute("type","CitationMention"));
		el.addAttribute(new Attribute("refType",type));
		el.addAttribute(new Attribute("charOffset",span.toString()));
		if (headSpan == null) {
			el.addAttribute(new Attribute("headOffset",surfaceElement.getHead().getSpan().toString()));
		} else {
			el.addAttribute(new Attribute("headOffset",headSpan.toString()));
		}
//		if (citation != null) 
//			el.addAttribute(new Attribute("citation",citation.getId()));
		if (sentiment != null) 
			el.addAttribute(new Attribute("sentiment",sentiment.toString()));
		if (context != null) {
			el.addAttribute(new Attribute("context",new SpanList(context).toString()));
		}
		if (subjectMatter != null) {
			el.addAttribute(new Attribute("subjectMatter",new SpanList(subjectMatter).toString()));
		}
		if (features != null) {
			for (String s: features.keySet()) {
				el.addAttribute(new Attribute(s,features.get(s).toString()));
			}
		}
		el.addAttribute(new Attribute("text",getText()));
		
		return el;
	}

	@Override
	public Ontology getOntology() {
//		return getReference();
		return null;
	}

	@Override
	public boolean ontologyEquals(Object obj) {
		if (obj instanceof CitationMention == false) return false;
		CitationMention ment = (CitationMention)obj;
		return (ment.getDocument().getId().equals(getDocument().getId()) && ment.getId().equals(getId()));
//		Citation cit = ment.getCitation();
//		return (cit.getReference().equals(getReference()));
	}
	
	public Reference getReference() {
//		return citation.getReference();
		return null;
	}
	
/*	public Citation getCitation() {
		return citation;
	}
	
	public void setCitation(Citation citation) {
		this.citation = citation;
	}*/
	
	public Sentiment getSentiment() {
		return sentiment;
	}

	public void setSentiment(Sentiment sentiment) {
		this.sentiment = sentiment;
	}
	
	@Override
	public int hashCode() {
		return 
	    ((id == null ? 89 : id.hashCode()) ^
	     (type  == null ? 97 : type.hashCode()) ^
	     (getText() == null ? 103: getText().hashCode()) ^
	     (span == null ? 119 : span.hashCode())); // ^
//	     (citation == null ? 139: citation.hashCode()));
	}
	
	/**
	 * Equality on the basis of type and mention equality.
	 */
	@Override
	public boolean equals(Object obj) {
		if (obj == null) return false;
		if (this == obj) return true;
		if (getClass() != obj.getClass()) return false;
		CitationMention at = (CitationMention)obj;
		return (id.equals(at.getId()) &&
				type.equals(at.getType()) &&
				getText().equals(at.getText()) &&
				span.equals(at.getSpan())); // &&
//				citation.equals(at.getCitation());
	}
	
	public Map<String, String> getMetaDataMap() {
		return metaDataMap;
	}
	public void setMetaDataMap(Map<String, String> metadataMap) {
		this.metaDataMap = metadataMap;
	}
	
	public void setMetaData(String key, String value) {
		if (metaDataMap == null) metaDataMap = new HashMap<String,String>();
		metaDataMap.put(key, value);
	}
	
	public String getMetaData(String key) {
		return metaDataMap.get(key);
	}
	
	  public Attributes getAttributes() {
		    if (attrs == null) {
		      attrs = new Attributes();
		    }
		    return attrs;
		  }
	  
	  public String toString() {
		  StringBuffer buf = new StringBuffer();
//		  buf.append(getDocument().getId() + "_" + getId() + "_" + getText() + "_" + sentiment.toString() + "\n");
		  buf.append(getDocument().getId() + "_" + getId() + "_" + getText() + "_" + sentiment.toString()+"_");
		  if (context == null) return buf.toString();
		  for (Span s: context) {
			  buf.append(" " + getDocument().getStringInSpan(s));
		  }
/*		  if (subjectMatter == null) return buf.toString();
		  for (Span s: subjectMatter) {
			  buf.append(" " + getDocument().getStringInSpan(s));
		  }*/
/*		  for (Span s: context) {
			  buf.append("\tCONTEXT:" + getDocument().getStringInSpan(s) + "\n");
		  }*/
		  return buf.toString();
	  }

}
