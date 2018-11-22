import xml.sax
import json
import argparse



class HyperPartisanGoldLabelHandler(xml.sax.ContentHandler):
    def __init__(self, out_file):
        xml.sax.ContentHandler.__init__(self)
        self.out_file = open(out_file,"w")
        self.article_id = ""
        self.hyperpartisan = ""
        self.bias = ""
        self.url = ""
        self.articles = []
    def startElement(self, name, attrs):
        self.current_data = name
        self.attrs = attrs
        if name == "article":            
            self.article_id  = attrs.getValue("id")
            self.hyperpartisan = attrs.getValue("hyperpartisan")
            self.bias = attrs.getValue("bias")
            self.url = attrs.getValue("url")
            self.labeled_by = attrs.getValue("labeled-by")
        
    def endElement(self, name):
        if name == 'article':
            article = {}
            article["article_id"] = self.article_id
            article["hyperpartisan"] = self.hyperpartisan
            article["bias"] = self.bias
            article["url"] = self.url
            self.articles.append(article)
            self.article_id  = ""
            self.hyperpartisan = ""
            self.bias = ""
            self.url = ""
            json.dump(article,self.out_file)
            self.out_file.write('\n')

            
class HyperpartisanNewsDataHandler(xml.sax.ContentHandler):
    def __init__(self, out_file):
        xml.sax.ContentHandler.__init__(self)
        self.articles = [] 
        self.out_file = open(out_file,"w")
        self.ps = []
        self.article_id = ""
        self.title = ""
        self.published_at = ""
        self.current_data = ""
        self.attrs = ""
        self.is_repeated = False
    def startElement(self, name, attrs):        
        self.current_data = name
        self.attrs = attrs
        if name == "article":
            self.article_id  = attrs.getValue("id")
            self.title = attrs.getValue("title")
            if "published-at" in attrs:
                self.published_at = attrs.getValue("published-at")
            else:
                self.published_at = ""                


    def characters(self, data):
        if self.current_data == "p":
            self.ps.append(data)
        if self.current_data == "a":
            if "href" in self.attrs and self.attrs["href"] != "":
                if self.is_repeated == False:
                    self.ps.append("[link] " +self.attrs["href"] + " [/link] "+data)
                else:
                    self.ps.append(data)
            else:
                self.ps.append(data)
            # to prevent from double time link adding
            if self.is_repeated == False:
                self.is_repeated = True
            else:
                self.is_repeated = False

        
    def endElement(self, name):
        if name == 'article':
            article = {}
            article["ps"] = self.ps
            article["article_id"] = self.article_id
            article["title"] = self.title
            article["published_at"] = self.published_at
            self.articles.append(article)
            self.ps = []
            self.article_id  = ""
            self.title = ""
            self.published_at = ""
            json.dump(article,self.out_file)
            self.out_file.write('\n')

def main():
    """Main method of this module."""

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--inputfile', dest='input_file',default="../data/articles-validation-20180831.xml",
                        help='name of the xml file')
    parser.add_argument('--inputfilelabel', dest='input_file_label',default="../data/ground-truth-validation-20180831.xml",
                        help='name of the xml file')
    parser.add_argument('--outputfile', dest='output_file',
                        default="../data/val.json",
                        help='name of the json file')
    parser.add_argument('--outputfilelabel', dest='output_file_label',
                        default="../data/val_label.json",
                        help='name of the json file')

    args = parser.parse_args()


    # inputs
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler = HyperpartisanNewsDataHandler(args.output_file)
    parser.setContentHandler( Handler )
    parser.parse(args.input_file)
    Handler.out_file.close()


    # labels
    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    Handler_Label = HyperPartisanGoldLabelHandler(args.output_file_label)
    parser.setContentHandler( Handler_Label )
    parser.parse(args.input_file_label)
    Handler_Label.out_file.close()

if __name__ == '__main__':
    main()
