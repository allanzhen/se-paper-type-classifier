# se-paper-type-classifier

## SE Topic of Interest: Technical Debt
Software engineering research takes many forms — controlled experiments, systematic literature reviews, tool papers, experience reports, and more. The mix of paper types a research community produces reflects how that community generates and validates knowledge, yet this distribution is rarely studied explicitly.
This project investigates the landscape of Technical Debt research as published in high-quality software engineering venues (CORE A and A* ranked). Using a fully automated NLP-based classification pipeline, each paper in the corpus is assigned a type label derived solely from its title and abstract. The resulting distribution is then analysed to characterise how the Technical Debt research community operates — whether it leans towards synthesis and description, empirical validation, or tool building — and whether that mix has shifted over time since the field emerged as a formal research area around 2010.
The core deliverable is a reproducible classification pipeline that accepts a paper title and abstract as input and returns a type label with a confidence score. Beyond its use in this study, the classifier is designed to be reusable across any SE topic and has been handed to the shared infrastructure group for packaging as a platform service.

