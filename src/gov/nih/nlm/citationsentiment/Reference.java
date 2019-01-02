package gov.nih.nlm.citationsentiment;

import java.util.ArrayList;
import java.util.List;

import org.w3c.dom.NamedNodeMap;
import org.w3c.dom.Node;
import org.w3c.dom.NodeList;

import gov.nih.nlm.ling.sem.Ontology;
import nu.xom.Attribute;
import nu.xom.Element;
import nu.xom.Elements;

/**
 * A representation of an article reference. 
 * 
 * @author Halil Kilicoglu
 *
 */
public class Reference implements Ontology {

	private String type;
	private String title;
	private String source;
	private String year;
	private String volume;
	private String firstPage;
	private String lastPage;
	private String pmcId;
	private String pubmedId;
	private String doi;
	private List<String> authors;
	
	public Reference()  {
		// TODO Auto-generated constructor stub
	}
	
	public Reference(Element el) {
		if (el.getLocalName().equals("citation") == false && el.getLocalName().equals("element-citation") == false && 
				el.getLocalName().equals("mixed-citation") == false) 
			throw new IllegalArgumentException("Not a citation XML element.");
		type = el.getAttributeValue("citation-type");
		if (type == null) 
			type = el.getAttributeValue("publication-type");
		title = el.getFirstChildElement("article-title").getValue();
		source = el.getFirstChildElement("source").getValue();
		year = el.getFirstChildElement("year").getValue();
		volume = el.getFirstChildElement("volume").getValue();
		firstPage = el.getFirstChildElement("fpage").getValue();
		lastPage = el.getFirstChildElement("lpage").getValue();
		Elements ids = el.getChildElements("pub-id");
		for (int i=0; i < ids.size(); i++) {
			Element idel = ids.get(i);
			String t = idel.getAttributeValue("pub-id-type");
			if (t.equals("pmid")) pubmedId = idel.getValue();
			else if (t.equals("doi")) doi = idel.getValue();
			else if (t.equals("pmc")) pmcId = idel.getValue();
		}
		Elements authorsEl = el.getChildElements("person-group");
		authors = new ArrayList<>();
		for (int i=0; i < authorsEl.size(); i++) {
			Element author = authorsEl.get(i);
			String avalue = author.getAttributeValue("person-group-type");
			if (avalue == null || avalue.equals("author")) {
				Elements names = author.getChildElements("name");
				for (int j=0; j < names.size(); j++) {
					Element name = names.get(i);
					String surname = name.getChildElements("surname").get(0).getValue();
					String given = name.getChildElements("given-names").get(0).getValue();
					String fullname = surname + ", " + given;
					authors.add(fullname);
				}
			}
		}
	}
	
	public Reference(Node el) {
		String name = el.getNodeName();
		if (name.equals("citation") == false && name.equals("element-citation") == false && name.equals("mixed-citation") == false) 
			throw new IllegalArgumentException("Not a citation XML element.");
        NamedNodeMap refAttributes = el.getAttributes();
        if (refAttributes.getNamedItem("citation-type") == null) {
			type = refAttributes.getNamedItem("publication-type").getNodeValue();
        } else
        	type = refAttributes.getNamedItem("citation-type").getNodeValue();

		NodeList children = el.getChildNodes();
		for (int l=0; l < children.getLength(); l++){
			Node child = children.item(l);
			NamedNodeMap childAtts = child.getAttributes(); 
			if (child.getNodeName().equals("article-title")) title = child.getTextContent();
			else if (child.getNodeName().equals("source")) source = child.getTextContent();
			else if (child.getNodeName().equals("year")) year = child.getTextContent();
			else if (child.getNodeName().equals("volume")) volume = child.getTextContent();
			else if (child.getNodeName().equals("fpage")) firstPage = child.getTextContent();
			else if (child.getNodeName().equals("lpage")) lastPage = child.getTextContent();
			else if (child.getNodeName().equals("pub-id")) {
				String t = childAtts.getNamedItem("pub-id-type").getNodeValue();
				if (t.equals("pmid")) pubmedId = child.getTextContent();
				else if (t.equals("doi")) doi = child.getTextContent();
				else if (t.equals("pmc")) pmcId = child.getTextContent();
			} 
			else if (child.getNodeName().equals("person-group")) {
				Node persontype = childAtts.getNamedItem("person-group-type");
				if (persontype == null || persontype.getNodeValue().equals("author")) {
					authors = new ArrayList<>();
					NodeList names = child.getChildNodes();
					for (int j=0; j < names.getLength(); j++) {
						Node namej = names.item(j);	
						NodeList subnames = namej.getChildNodes();
						String surname = "";
						String given = "";
						for (int k=0; k < subnames.getLength(); k++) {
							Node sname = subnames.item(k);
							if (sname.getNodeName().equals("surname")) surname = sname.getTextContent();
							if (sname.getNodeName().equals("given-names")) given = sname.getTextContent();
						}
						StringBuffer buf = new StringBuffer();
						if (surname.equals("") == false) {
							buf.append(surname);
							buf.append(",");
						}
						if (given.equals("") == false) buf.append(given);
						String fullname = buf.toString();
						if (fullname.equals("") == false) authors.add(fullname);
					}
				}
			} 
		}
	}

	public String getType() {
		return type;
	}

	public String getTitle() {
		return title;
	}

	public String getSource() {
		return source;
	}

	public String getYear() {
		return year;
	}

	public String getVolume() {
		return volume;
	}

	public String getFirstPage() {
		return firstPage;
	}

	public String getLastPage() {
		return lastPage;
	}

	public String getPmcId() {
		return pmcId;
	}

	public String getPubmedId() {
		return pubmedId;
	}

	public String getDoi() {
		return doi;
	}

	public List<String> getAuthors() {
		return authors;
	}
	
	public boolean equals(Object obj) {
		if (obj == null) return false;
		if (this == obj) return true;
		if (getClass() != obj.getClass()) return false;
		Reference ref = (Reference)obj;
		String refPubmedId = ref.getPubmedId();
		String refPmcId = ref.getPmcId();
		String refDoi = ref.getDoi();
		return ((pmcId != null && refPmcId != null && pmcId.equals(refPmcId)) ||
				(pubmedId != null && refPubmedId != null && pubmedId.equals(refPubmedId)) ||
				(doi != null && refDoi != null && doi.equals(refDoi)));
	}
	
	public int hashCode() {
		if (pmcId != null) return pmcId.hashCode();
		if (pubmedId != null) return pubmedId.hashCode();
		if (doi != null) return doi.hashCode();
	    return ((title == null ? 119 : title.hashCode()) ^
	    		(source == null ? 89: source.hashCode()) ^
	    		(firstPage == null ? 59 : firstPage.hashCode()) ^
	    		(lastPage == null ? 79 : lastPage.hashCode()));
	}
	
	public Element toXml() {
		Element el = new Element("Reference");
		if (pubmedId != null) el.addAttribute(new Attribute("pubmedId",pubmedId));
		if (pmcId != null) el.addAttribute(new Attribute("pmcId",pmcId));
		if (doi != null) el.addAttribute(new Attribute("doi",doi));
		if (type != null) el.addAttribute(new Attribute("type",type));
		if (title != null) el.addAttribute(new Attribute("title",title));
		if (source != null) el.addAttribute(new Attribute("source",source));
		if (year != null) el.addAttribute(new Attribute("year",year));
		if (volume != null) el.addAttribute(new Attribute("volume",volume));
		if (firstPage != null) el.addAttribute(new Attribute("firstPage",firstPage));
		if (lastPage != null) el.addAttribute(new Attribute("lastPage",lastPage));
		if (authors != null && authors.size() > 0) el.addAttribute(new Attribute("authors",String.join(";", authors))); 
		return el;
	}
}
