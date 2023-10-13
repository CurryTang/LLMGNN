configs = {
    "cora": {
        "zero-shot": {
            "object-cat": "Paper",
            "question": "What's the category of this paper",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        },
        "few-shot": {
            "examples": [
                ("Stochastic pro-positionalization of non-determinate background knowledge. : It is a well-known fact that propositional learning algorithms require \"good\" features to perform well in practice. So a major step in data engineering for inductive learning is the construction of good features by domain experts. These features often represent properties of structured objects, where a property typically is the occurrence of a certain substructure having certain properties. To partly automate the process of \"feature engineering\", we devised an algorithm that searches for features which are defined by such substructures. The algorithm stochastically conducts a top-down search for first-order clauses, where each clause represents a binary feature. It differs from existing algorithms in that its search is not class-blind, and that it is capable of considering clauses (\"context\") of almost arbitrary length (size). Preliminary experiments are favorable, and support the view that this approach is promising.", "rule_learning")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("Stochastic pro-positionalization of non-determinate background knowledge. : It is a well-known fact that propositional learning algorithms require \"good\" features to perform well in practice. So a major step in data engineering for inductive learning is the construction of good features by domain experts. These features often represent properties of structured objects, where a property typically is the occurrence of a certain substructure having certain properties. To partly automate the process of \"feature engineering\", we devised an algorithm that searches for features which are defined by such substructures. The algorithm stochastically conducts a top-down search for first-order clauses, where each clause represents a binary feature. It differs from existing algorithms in that its search is not class-blind, and that it is capable of considering clauses (\"context\") of almost arbitrary length (size). Preliminary experiments are favorable, and support the view that this approach is promising.", 
                    "[{\"answer\": \"rule_learning\", \"confidence\": 60}, {\"answer\": \"probabilistic_methods\", \"confidence\": 30}, {\"answer\": \"theory\", \"confidence\": 10}]"
 )
            ]
        }
    }, 
    "citeseer": {
        "zero-shot": {
            "object-cat": "Paper",
            "question": "What's the category of this paper",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        }, 
        "few-shot": {
            "examples": [
                ("Argument in Multi-Agent Systems Multi-agent systems research is concerned both with the modelling of human and animal societies and with the development of principles for the design of practical distributed information management systems. This position paper will, rather than examine the various dierences in perspective within this area of research, discuss issues of communication and commitment that are of interest to multi-agent systems research in general. 1 Introduction A computational society is a collection of autonomous agents that are loosely dependent upon each other. The intentional stance [12] is often taken in describing the state of these agents. An agent may have beliefs, desires, intentions, and it may adopt a role or have relationships with others. Thus, multi-agent systems (MAS) as with most AI research is signi cantly inuenced, at least in its vocabulary, by philosophy and cognitive psychology. 1 So, what's the point? Computational societies are developed for two primary reasons:  Mode...", "agents")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("Argument in Multi-Agent Systems Multi-agent systems research is concerned both with the modelling of human and animal societies and with the development of principles for the design of practical distributed information management systems. This position paper will, rather than examine the various dierences in perspective within this area of research, discuss issues of communication and commitment that are of interest to multi-agent systems research in general. 1 Introduction A computational society is a collection of autonomous agents that are loosely dependent upon each other. The intentional stance [12] is often taken in describing the state of these agents. An agent may have beliefs, desires, intentions, and it may adopt a role or have relationships with others. Thus, multi-agent systems (MAS) as with most AI research is signi cantly inuenced, at least in its vocabulary, by philosophy and cognitive psychology. 1 So, what's the point? Computational societies are developed for two primary reasons:  Mode...", 
                "[{\"answer\": \"agents\", \"confidence\": 60}, {\"answer\": \"artificial intelligence\", \"confidence\": 30}, {\"answer\": \"human computer interaction\", \"confidence\": 10}]")
            ]
        }
    },
    "pubmed": {
        "zero-shot": {
            "object-cat": "Paper",
            "question": "What's the category of this paper",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        }, 
        "few-shot": {
            "examples": [
                ("Title: Retinal metabolic abnormalities in diabetic mouse: comparison with diabetic rat.\nAbstract: PURPOSE: Dogs and rats are commonly used to examine the pathogenesis of diabetic retinopathy, but mouse is sparingly studied as an animal model of diabetic retinopathy. In this study metabolic abnormalities, postulated to contribute to the development of retinopathy in diabetes, are investigated in the retina of mice diabetic or galactose-fed for 2 months, and are compared to those obtained from hyperglycemic rats. METHODS: Diabetes was induced in mice (C57BL/6) and rats (Sprague Dawley) by alloxan injection, and experimental galactosemia by feeding normal animals diets supplemented with 30% galactose. After 2 months of hyperglycemia, levels of lipid peroxides, glutathione, nitric oxides and sorbitol, and activities of protein kinase C and (Na-K)-ATPase were measured in the retina. RESULTS: Two months of diabetes or experimental galactosemia in mice increased retinal oxidative stress, PKC activity and nitric oxides by 40-50% and sorbitol levels by 3 folds, and these abnormalities were similar to those observed in the retina of rats hyperglycemic for 2 months. CONCLUSIONS: Metabolic abnormalities, which are postulated to play important role in the development of diabetic retinopathy in other animal models, are present in the retina of diabetic mice, and the level of metabolic abnormalities is very comparable between mice and rats. Thus, mouse seems to be a promising animal model to study the pathogenesis of diabetic retinopathy.", "diabetes mellitus, experimental")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("Title: Retinal metabolic abnormalities in diabetic mouse: comparison with diabetic rat.\nAbstract: PURPOSE: Dogs and rats are commonly used to examine the pathogenesis of diabetic retinopathy, but mouse is sparingly studied as an animal model of diabetic retinopathy. In this study metabolic abnormalities, postulated to contribute to the development of retinopathy in diabetes, are investigated in the retina of mice diabetic or galactose-fed for 2 months, and are compared to those obtained from hyperglycemic rats. METHODS: Diabetes was induced in mice (C57BL/6) and rats (Sprague Dawley) by alloxan injection, and experimental galactosemia by feeding normal animals diets supplemented with 30% galactose. After 2 months of hyperglycemia, levels of lipid peroxides, glutathione, nitric oxides and sorbitol, and activities of protein kinase C and (Na-K)-ATPase were measured in the retina. RESULTS: Two months of diabetes or experimental galactosemia in mice increased retinal oxidative stress, PKC activity and nitric oxides by 40-50% and sorbitol levels by 3 folds, and these abnormalities were similar to those observed in the retina of rats hyperglycemic for 2 months. CONCLUSIONS: Metabolic abnormalities, which are postulated to play important role in the development of diabetic retinopathy in other animal models, are present in the retina of diabetic mice, and the level of metabolic abnormalities is very comparable between mice and rats. Thus, mouse seems to be a promising animal model to study the pathogenesis of diabetic retinopathy.", 
                "[{\"answer\": \"diabetes mellitus experimental\", \"confidence\": 85},{\"answer\": \"diabetes mellitus type 1\", \"confidence\": 10},{\"answer\": \"diabetes mellitus type 2\", \"confidence\": 5}]")    
            ]
        }
    },
    "arxiv": {
        "zero-shot": {
            "object-cat": "Paper",
            "question": "Which arxiv CS-subcategory does this paper belong to?",
            "answer-format": "Give 3 likely arxiv CS-subcategories from most to least likely in the form of arXiv CS sub-categories like \"cs.XX\" together with a confidence ranging from 0 to 100, the sum of confidence should be 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}, ...]"
        }, 
        "few-shot": {
            "examples": [
                ("evasion attacks against machine learning at test time In security-sensitive applications, the success of machine learning depends on a thorough vetting of their resistance to adversarial data. In one pertinent, well-motivated attack scenario, an adversary may attempt to evade a deployed system at test time by carefully manipulating attack samples. In this work, we present a simple but effective gradient-based approach that can be exploited to systematically assess the security of several, widely-used classification algorithms against evasion attacks. Following a recently proposed framework for security evaluation, we simulate attack scenarios that exhibit different risk levels for the classifier by increasing the attacker's knowledge of the system and her ability to manipulate attack samples. This gives the classifier designer a better picture of the classifier performance under evasion attacks, and allows him to perform a more informed model selection (or parameter setting). We evaluate our approach on the relevant security task of malware detection in PDF files, and show that such systems can be easily evaded. We also sketch some countermeasures suggested by our analysis.", "cs.CR")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("evasion attacks against machine learning at test time In security-sensitive applications, the success of machine learning depends on a thorough vetting of their resistance to adversarial data. In one pertinent, well-motivated attack scenario, an adversary may attempt to evade a deployed system at test time by carefully manipulating attack samples. In this work, we present a simple but effective gradient-based approach that can be exploited to systematically assess the security of several, widely-used classification algorithms against evasion attacks. Following a recently proposed framework for security evaluation, we simulate attack scenarios that exhibit different risk levels for the classifier by increasing the attacker's knowledge of the system and her ability to manipulate attack samples. This gives the classifier designer a better picture of the classifier performance under evasion attacks, and allows him to perform a more informed model selection (or parameter setting). We evaluate our approach on the relevant security task of malware detection in PDF files, and show that such systems can be easily evaded. We also sketch some countermeasures suggested by our analysis.", 
                "[{\"answer\": \"cs.CR\", \"confidence\": 85},{\"answer\": \"cs.LG\", \"confidence\": 10},{\"answer\": \"cs.CV\", \"confidence\": 5}]")
            ]
        }
    },
    "products": {
        "zero-shot": {
            "object-cat": "Product",
            "question": "What's the category of this product",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        }, 
        "few-shot": {
            "examples": [
                ("Clara Clark Premier 1800 Collection 4pc Bed Sheet Set - Full (Double) Size, White", 'home & kitchen')
            ]
        },
        "few-shot-2": {
            "examples": [
                ("Clara Clark Premier 1800 Collection 4pc Bed Sheet Set - Full (Double) Size, White",
                "[{\"answer\": \"home & kitchen\", \"confidence\": 90},{\"answer\": \"grocery & gourmet food\", \"confidence\": 10},{\"answer\": \"health & personal care\", \"confidence\": 5}]")
            ]
        }
    },
    "wikics": {
        "zero-shot": {
            "object-cat": "Wiki Article",
            "question": "What's the wiki category of this article",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        }, 
        "few-shot": {
            "examples": [
                ("Twilio:twilio twilio cloud communications platform service cpaas company based san francisco california twilio allows software developers programmatically make receive phone calls send receive text messages perform communication functions using web service apis twilio founded 2008 jeff lawson evan cooke john wolthuis originally based seattle washington san francisco california twilio first major press coverage november 2008 result application built jeff lawson rickroll people investor dave mcclure used techcrunch founder editor michael arrington prank days later november 20 2008 company launched twilio voice api make receive phone calls completely hosted cloud twilio text messaging api released february 2010 sms shortcodes released public beta july 2011 twilio raised approximately 103 million venture capital growth funding twilio received first round seed funding march 2009 undisclosed amount rumored around 250,000 mitch kapor founders fund dave mcclure david g. cohen chris sacca manu kumar k9 ventures jeff fluhr twilio first round funding led union square ventures 3.7 million second b round funding 12 million led bessemer venture partners twilio received 17 million series c round december 2011 bessemer venture partners union square ventures july 2013 twilio received another 70 million redpoint ventures draper fisher jurvetson dfj bessemer venture partners july 2015 twilio raised 130 million series e fidelity rowe price altimeter capital management arrowpoint partners addition amazon salesforce twilio known use platform evangelism acquire customers early example groupme founded may 2010 techcrunch disrupt hackathon uses twilio text messaging product facilitate group chat raised 10.6 million venture funding january 2011 following success techcrunch disrupt hackathon seed accelerator 500 startups announced twilio fund 250,000", "distributed computing architecture")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("Twilio:twilio twilio cloud communications platform service cpaas company based san francisco california twilio allows software developers programmatically make receive phone calls send receive text messages perform communication functions using web service apis twilio founded 2008 jeff lawson evan cooke john wolthuis originally based seattle washington san francisco california twilio first major press coverage november 2008 result application built jeff lawson rickroll people investor dave mcclure used techcrunch founder editor michael arrington prank days later november 20 2008 company launched twilio voice api make receive phone calls completely hosted cloud twilio text messaging api released february 2010 sms shortcodes released public beta july 2011 twilio raised approximately 103 million venture capital growth funding twilio received first round seed funding march 2009 undisclosed amount rumored around 250,000 mitch kapor founders fund dave mcclure david g. cohen chris sacca manu kumar k9 ventures jeff fluhr twilio first round funding led union square ventures 3.7 million second b round funding 12 million led bessemer venture partners twilio received 17 million series c round december 2011 bessemer venture partners union square ventures july 2013 twilio received another 70 million redpoint ventures draper fisher jurvetson dfj bessemer venture partners july 2015 twilio raised 130 million series e fidelity rowe price altimeter capital management arrowpoint partners addition amazon salesforce twilio known use platform evangelism acquire customers early example groupme founded may 2010 techcrunch disrupt hackathon uses twilio text messaging product facilitate group chat raised 10.6 million venture funding january 2011 following success techcrunch disrupt hackathon seed accelerator 500 startups announced twilio fund 250,000", 
                "[{\"answer\": \"distributed computing architecture\": \"confidence\": 70}, {\"answer\": \"web technology\", \"confidence\": 20}, {\"answer\": \"internet protocols\", \"confidence\": 10}]")
            ]
        }  
    }, 
    "tolokers": {
        "zero-shot": {
            "object-cat": "Your task is to determine whether this user has been banned on a crowdsourcing platform. You are given four attributes ranging from 0 to 1 representing the ratio of each user's approved, skipped, expired, and rejected tasks. Moreover, you are given three attributes that describe users' education and English proficiency. User attributes",
            "question": "Whether this user has been banned",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        }, 
        "few-shot": {
            
        }
    },
    "20newsgroup": {
        "zero-shot": {
            "object-cat": "News",
            "question": "What's the category of this news",
            "answer-format": "Output your answer together with a confidence ranging from 0 to 100, in the form of a list of python dicts like [{\"answer\":<answer_here>, \"confidence\": <confidence_here>}]"
        },
        "few-shot": {
            "examples": [
                ("organization post office carnegie mellon pittsburgh pa lines nntppostinghost po andrewcmuedu i am sure some bashers of pens fans are pretty confused about the lack of any kind of posts about the recent pens massacre of the devils actually i am bit puzzled too and a bit relieved however i am going to put an end to nonpittsburghers relief with a bit of praise for the pens man they are killing those devils worse than i thought jagr just showed you why he is much better than his regular season stats he is also a lot fo fun to watch in the playoffs bowman should let jagr have a lot of fun in the next couple of games since the pens are going to beat the pulp out of jersey anyway i was very disappointed not to see the islanders lose the final regular season game pens rule", "news about hockey.")
            ]
        },
        "few-shot-2": {
            "examples": [
                ("organization post office carnegie mellon pittsburgh pa lines nntppostinghost po andrewcmuedu i am sure some bashers of pens fans are pretty confused about the lack of any kind of posts about the recent pens massacre of the devils actually i am bit puzzled too and a bit relieved however i am going to put an end to nonpittsburghers relief with a bit of praise for the pens man they are killing those devils worse than i thought jagr just showed you why he is much better than his regular season stats he is also a lot fo fun to watch in the playoffs bowman should let jagr have a lot of fun in the next couple of games since the pens are going to beat the pulp out of jersey anyway i was very disappointed not to see the islanders lose the final regular season game pens rule", 
                "[{\"answer\": \"news about baseball.\", \"confidence\": 90},{\"answer\": \"news about miscellaneous political topics.\", \"confidence\": 7},{\"answer\": \"news about miscellaneous religious topics.\", \"confidence\": 3}]")
            ]
        }
    }
}