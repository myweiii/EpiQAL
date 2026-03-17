API_KEY = ""
OPENAI_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
GENERATION_MODEL_TYPE = "LOCAL"
GENERATION_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SIMILARITY_ENCODER_NAME = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

EKGDONS_FILE_PATH = ""
IBKH_FILE_PATH = ""

DATA_PATH = ""
RESULT_FILE_PATH = "./output"
MAX_GENERATE_ATTEMPT = 2
BATCH_SIZE = 32
GENERATION_TEMPRATURE = 0
GENERATION_MAX_TOKENS = 10240
GENERATION_TOP_P = 0.95
SIMILARITY_SEARCH_K = 5
KG_SIMILARITY_THRES = 0.9

CHECK_TIME = 3
CHECK_TEMPRATURE = 1
CHECK_MAX_TOKENS = 10240
CHECK_TOP_P = 0.95

# (URL, API_KEY, MODEL_NAME, BATCH_SIZE)
# (MODEL_NAME, QUANTIZATION, BATCH_SIZE)
OPENAI_API_KEY = ""
DPSK_API_KEY = ""
CHECK_MODEL_GROUP = {"API": [#(OPENAI_BASE_URL, API_KEY, "qwen3", 4),
                             #(OPENAI_BASE_URL, API_KEY, "gemma3", 4),
                             #(OPENAI_BASE_URL, API_KEY, "llama3-sdsc", 4),
                             ("https://api.openai.com/v1", OPENAI_API_KEY, "gpt-5-mini", 12),
                             ("https://api.deepseek.com", DPSK_API_KEY, "deepseek-reasoner", 12)
                             ],
                     "LOCAL": [("zai-org/GLM-4.5-Air", "bitsandbytes", 8),
                               #("mistralai/Mistral-Large-Instruct-2411", "bitsandbytes", 8),
                               #("meta-llama/Llama-3.3-70B-Instruct", "bitsandbytes", 8),
                               #("Qwen/Qwen3-30B-A3B-Instruct-2507", None, 32),
                               #("Qwen/Qwen3-Next-80B-A3B-Instruct", "bitsandbytes", 8),
                                ]}

CHECK_MODEL_NUM = (len(CHECK_MODEL_GROUP["API"]) + len(CHECK_MODEL_GROUP["LOCAL"])) * CHECK_TIME
CHECK_VOTE_THRES = 6/9
HUMAN_REVIEW_THRES = 5/9



JUDGE_MODEL_GROUP = {"API": [#(OPENAI_BASE_URL, API_KEY, "qwen3", 4),
                             #(OPENAI_BASE_URL, API_KEY, "gemma3", 4),
                             #(OPENAI_BASE_URL, API_KEY, "llama3-sdsc", 4),
                             ("https://api.openai.com/v1", OPENAI_API_KEY, "gpt-5-mini", 48),
                             ("https://api.deepseek.com", DPSK_API_KEY, "deepseek-reasoner", 48),
                             ],
                     "LOCAL": [#("zai-org/GLM-4.5-Air", "bitsandbytes", 8),
                                ("microsoft/Phi-4-mini-instruct", None, 64),
                                #("meta-llama/Llama-3.2-3B-Instruct", None, 64),
                                #("meta-llama/Llama-3.1-8B-Instruct", None, 32),
                                #("meta-llama/Llama-3.3-70B-Instruct", "bitsandbytes", 12),
                                #("mistralai/Mistral-7B-Instruct-v0.3", None, 32),
                                #("Qwen/Qwen3-8B", None, 32),
                                #("Qwen/Qwen3-30B-A3B-Instruct-2507", None, 16),
                                ("Qwen/Qwen3-32B", None, 24),
                                #("Qwen/Qwen3-Next-80B-A3B-Instruct", "bitsandbytes", 8),
                                #("tiiuae/falcon-40b", "bitsandbytes", 16),
                                ]}

JUDGE_TEMPRATURE = 0
JUDGE_MAX_TOKENS = 10240
JUDGE_TOP_P = 0.95


MAX_REVISION_TIMES = 3
DEFINITION_SEARCH_URL = "https://google.serper.dev/search"
DEFINITION_SEARCH_API_KEY = ''
DEFINITION_SEARCH_MAX_SNIPPET = 6

ALPHA = 0.3
DIFFICULTY_THRESHOLD = 0.2

### Local Inference
KG_MODEL_TYPE = "LOCAL"
KG_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507" #"meta-llama/Llama-3.3-70B-Instruct"
KG_LOCAL_QUANTIZATION = None #"bitsandbytes"
MAX_KG_ITEMS_PER_KEY = 50
KG_BATCH_SIZE = 16#8
LOCAL_TENSOR_PARALLEL_SIZE = 1
LOCAL_QUANTIZATION = None   #"bitsandbytes"
LOCAL_MAX_TOKENS = 20480
LOCAL_MAX_MODEL_LEN = 124000



EVAL_MODEL_GROUP = {"API": [#(OPENAI_BASE_URL, API_KEY, "qwen3", 8),
                             #(OPENAI_BASE_URL, API_KEY, "gemma3", 4),
                             #(OPENAI_BASE_URL, API_KEY, "llama3-sdsc", 4),
                             ("https://api.openai.com/v1", OPENAI_API_KEY, "gpt-4.1-nano", 36),
                             ("https://api.openai.com/v1", OPENAI_API_KEY, "gpt-4o-mini", 36),
                             ("https://api.openai.com/v1", OPENAI_API_KEY, "gpt-5-mini", 36),
                             ("https://api.deepseek.com", DPSK_API_KEY, "deepseek-reasoner", 36)
                             ],
                     "LOCAL": [("zai-org/GLM-4.5-Air", "bitsandbytes", 16),
                                ("microsoft/Phi-4-mini-instruct", None, 128),
                                ("meta-llama/Llama-3.2-3B-Instruct", None, 128),
                                ("meta-llama/Llama-3.1-8B-Instruct", None, 64),
                                ("meta-llama/Llama-3.3-70B-Instruct", "bitsandbytes", 16),
                                ("mistralai/Mistral-7B-Instruct-v0.3", None, 64),
                                ("mistralai/Mistral-Large-Instruct-2411", "bitsandbytes", 16),
                                ("Qwen/Qwen3-8B", None, 64),
                                ("Qwen/Qwen3-30B-A3B-Instruct-2507", None, 32),
                                ("Qwen/Qwen3-32B", None, 32),
                                #("Qwen/Qwen3-Next-80B-A3B-Instruct", "bitsandbytes", 12),
                                #("tiiuae/falcon-40b", "bitsandbytes", 16),
                                ]}



QUESTION_CLASS = [{"Index": "1",
                    "Class": "Surveillance & Descriptive Epidemiology",
                    "Description": "Describes population occurrence from routine data (rates, time-place-person, aberration signals) and basic system performance, without causal analysis or forecasting."},
                    {"Index": "2",
                    "Class": "Outbreak Investigation & Field Response",
                    "Description": "Handles outbreak-specific confirmation, field case definitions, line lists, attack rates and curves, chain/source hypotheses, and immediate control with situation reports."},
                    {"Index": "3",
                    "Class": "Determinants & Exposures",
                    "Description": "Explains how exposure arises across settings, covering behavioral, environmental, occupational, and social determinants; delineates canonical transmission routes and contact structures; interprets exposure-response with attention to measurement methods, units, detection limits, and thresholds; and situates risks within One Health interfaces involving reservoirs and vectors."},
                    {"Index": "4",
                    "Class": "Susceptibility & Immunity",
                    "Description": "Describes who is susceptible and why, linking serologic measures to correlates of protection; evaluates effectiveness after vaccination or prior infection and its waning with reinfection, hybrid immunity, and variant escape, including the effects of vaccine dose number and intervals; and assesses severity risk using clinical and contextual prognostic factors."},
                    {"Index": "5",
                    "Class": "Modeling, Methods & Evaluation",
                    "Description": "Provides analytical methods: transmission modeling and inference, real-time debiasing of surveillance data, study design and causal effects, measurement/bias handling, and program performance/burden evaluation."},
                    {"Index": "6",
                    "Class": "Projections & Forecasts",
                    "Description": "Produces forward-looking forecasts and scenarios, evaluates and combines models, and supports decision making; it does not reconstruct recent under-reported data."}
                    ]

QUESTION_TOPIC = {"1": [{
                    "Index": "1",
                    "Topic": "Frequency measures & standardization",
                    "Description": "Defines prevalence, incidence, person-time, and applies standardization to make rates comparable."
                    },
                    {
                    "Index": "2",
                    "Topic": "Time-Place-Person patterns, seasonality & clustering",
                    "Description": "Describes temporal trends, spatial distribution, and demographic profiles using routine population surveillance."
                    },
                    {
                    "Index": "3",
                    "Topic": "Aberration & Outbreak detection",
                    "Description": "Builds statistical baselines and thresholds to flag unusual increases in counts, rates, or positivity; focuses on signal detection rather than source attribution."
                    },
                    {
                    "Index": "4",
                    "Topic": "System Performance, Deduplication & Record Linkage",
                    "Description": "Assesses sensitivity, timeliness, and completeness; manages deduplication and linkage across multiple data sources."
                    }],
                "2": [{
                    "Index": "1",
                    "Topic": "Diagnostic verification, field case definitions & line lists",
                    "Description": "Confirms the pathogen, applies field case definitions, and builds/cleans line lists."
                    },
                    {
                    "Index": "2",
                    "Topic": "Event-specific attack rates & epidemic curves",
                    "Description": "Quantifies spread in defined groups and interprets epidemic curves for the event."
                    },
                    {
                    "Index": "3",
                    "Topic": "Outbreak hypothesis mapping & Source Attribution",
                    "Description": "Links cases by time, place, and shared exposures to identify the most likely sources and transmission chain, integrating line lists, environmental sampling, traceback, and genomic evidence."
                    },
                    {
                    "Index": "4",
                    "Topic": "Immediate control & situation reporting",
                    "Description": "Implements urgent measures and documents current status with concise situation reports."
                    }],
                "3": [{
                    "Index": "1",
                    "Topic": "Contextual determinants of exposure",
                    "Description": "Integrates individual behaviours with environmental, occupational, and social/structural conditions that shape exposure probability and inequities."
                    },
                    {
                    "Index": "2",
                    "Topic": "Transmission modes & contact patterns",
                    "Description": "Describes general routes of spread and population contact structures across settings."
                    },
                    {
                    "Index": "3",
                    "Topic": "Exposure-response interpretation",
                    "Description": "Specifies the exposure metric (what/how measured), determines whether values are above/below assay limits and thresholds, and interprets exposure to infection/severity/transmissibility patterns as reported in the passage."
                    },
                    {
                    "Index": "4",
                    "Topic": "Zoonotic/One Health interfaces, reservoirs & vectors",
                    "Description": "Identifies animal reservoirs, vectors, and human-animal-environment interfaces where spillover can occur."
                    }],
                "4": [{
                    "Index": "1",
                    "Topic": "Susceptibility stratification & special populations",
                    "Description": "Identifies groups more susceptible to infection based on demographic/clinical traits and setting-specific contexts."
                    },
                    {
                    "Index": "2",
                    "Topic": "Serology & correlates of protection",
                    "Description": "Estimates seroprevalence and relates immune markers to protection thresholds and population-level immunity."
                    },
                    {
                    "Index": "3",
                    "Topic": "Protection effectiveness, waning, reinfection & immune escape",
                    "Description": "Describes protection after vaccination or prior infection, its change over time, risks of reinfection, hybrid immunity, and variant-related escape. Considers how vaccine dose number and dose intervals influence vaccine effectiveness and its waning over time."
                    },
                    {
                    "Index": "4",
                    "Topic": "Severity risk & prognostic factors",
                    "Description": "Assesses risk of severe outcomes conditional on infection and stratifies prognosis by host factors."
                    }],
                "5": [{
                    "Index": "1",
                    "Topic": "Transmission modeling & inference",
                    "Description": "Uses mechanistic or statistical models to estimate transmission parameters and infer transmission patterns."
                    },
                    {
                    "Index": "2",
                    "Topic": "Real-Time Debiasing & Delay Adjustment",
                    "Description": "Reconstructs recent incidence by adjusting for reporting delays, right truncation, and under-ascertainment."
                    },
                    {
                    "Index": "3",
                    "Topic": "Study design & causal effects",
                    "Description": "Selects designs and identification strategies and defines effect measures for causal estimation."
                    },
                    {
                    "Index": "4",
                    "Topic": "Measurement & bias handling",
                    "Description": "Addresses measurement validity, misclassification and measurement error, confounding and selection, generalizability, survey weighting, and sample size."
                    },
                    {"Index": "5",
                        "Topic": "Program performance & impact evaluation",
                        "Description": "Assesses coverage and implementation fidelity, audits routine data quality, evaluates real-world effectiveness, and estimates disease burden."
                    }],
                "6": [{
                    "Index": "1",
                    "Topic": "Near-term forecasting",
                    "Description": "Produces short-horizon probabilistic forecasts for upcoming values and quantifies forecast uncertainty."
                    },
                    {
                    "Index": "2",
                    "Topic": "Scenario projections",
                    "Description": "Projects future trajectories under stated assumptions about policy, behaviour, or immunity."
                    },
                    {
                    "Index": "3",
                    "Topic": "Forecast evaluation & model combination",
                    "Description": "Assesses the quality of epidemiological forecasts using proper scoring rules, calibration, and sharpness diagnostics; and develops or applies methods to combine multiple forecasting models to improve predictive accuracy, stability, and robustness across contexts."
                    },
                    {
                    "Index": "4",
                    "Topic": "Decision-oriented forecasting & risk communication",
                    "Description": "Maps forecast probabilities to operational thresholds or cost-loss trade-offs and communicates uncertainty for decision-making."
                    }]
            }

