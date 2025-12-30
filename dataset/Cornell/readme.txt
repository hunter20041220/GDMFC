===== Presentation =====
This dataset have been downloaded from http://www.cs.umd.edu/projects/linqs/projects/lbc/
		
date: February, 2012


===== Description =====
The archive contains 195 documents over the 5 labels (student,project,course,staff,faculty). It is made of 4 views (content,inbound,outbound,cites) on the same documents. The documents are described by 1703 words in the content view, and by the 569 links between them in the inbound, outbound and cites views.

===== Files =====
All the files are encoded in UTF8.

cornell_content.mtx -- 
	the documents-words matrix, containing 0/1 values indicating absence/presence of a word in a document, in the Matrix Market coordinate format (sparse).

cornell_inbound.mtx -- 
	the matrix indicating by 0/1 values the inbound links between documents, in the Matrix Market coordinate format (sparse).

cornell_outbound.mtx -- 
	the matrix indicating by 0/1 values the outbound links between documents, in the Matrix Market coordinate format (sparse). It is the transpose of cornell_inbound.mtx.

cornell_cites.mtx -- 
	the matrix of the number of citation links between documents, in the Matrix Market coordinate format (sparse). It is the sum of cornell_inbound.mtx and cornell_outbound.mtx.

documents-mapping.txt -- 
	the mapping between the rows of the matrix and the id of the document in the original collection.

cornell.txt --
	contains the list of the affectations of the documents to a topic.

labels.txt --
	contains the list of the different labels, in the order of the affectations found in cornell_act.txt