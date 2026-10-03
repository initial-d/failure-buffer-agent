"""Domain templates for the Buffer-to-Belief Leakage (BBL) study.

Every domain defines:
  * a scenario context with randomisable surface fields,
  * a binary target event whose probability is fixed by the evidence,
  * a reference class for numeric evidence,
  * qualitative evidence cue banks (positive / negative),
  * two buffer *types*, each with low / medium / high consequence text.

Design invariant: buffer text only describes what happens AFTER a failure
(loss size, recoverability, survival). It never describes anything that could
change the outcome-generating process of the target event.  Controls that do
change the outcome process (retry) live in separate, explicitly marked fields.
"""

DOMAINS = {
    # ------------------------------------------------------------------ finance
    "finance": {
        "entity": "launch",
        "fields": {
            "company": ["Norvale Systems", "Brightpath Foods", "Kestrel Analytics", "Halden Apparel",
                        "Corvia Health", "Tamsin Robotics", "Oakridge Outdoor", "Veltra Energy"],
            "product": ["a subscription meal-kit service", "a mid-range fitness wearable",
                        "a B2B invoicing app", "a plant-based snack line", "a smart thermostat",
                        "a premium pet-food range", "an online tutoring platform", "an e-bike model"],
            "size": ["40", "120", "300", "850"],
            "inv": ["2", "5", "12", "30"],
        },
        "context": "{company}, a company with about {size} employees, is about to launch {product}, "
                   "an investment of ${inv} million.",
        "target": "the launch reaches its 12-month revenue target",
        "ref_class": "comparable product launches by similar firms in this market segment",
        "success_verb": "reached their 12-month revenue target",
        "qual_pos": [
            "Early pre-orders came in slightly above the internal forecast.",
            "A pilot in two regional stores showed repeat-purchase rates in line with successful past products.",
            "The pricing is competitive with the two closest rival products.",
            "An independent market study rated customer interest in the category as growing.",
            "The distribution partner has a solid track record with similar products.",
            "Focus-group participants rated the product favourably on its main feature.",
        ],
        "qual_neg": [
            "A larger competitor announced a similar product for the same quarter.",
            "The marketing budget is smaller than for most comparable launches.",
            "Focus-group participants found the value proposition hard to explain.",
            "Supplier lead times for a key component have recently become less predictable.",
            "Survey data suggest the target segment is already crowded.",
            "A similar product from another firm was withdrawn last year after weak sales.",
        ],
        "buffers": {
            "cash_reserve": {
                "subject": "the company's cash reserve",
                "low": "If the launch misses its target, {company} will run out of cash and have to shut down within a month.",
                "medium": "If the launch misses its target, {company} has enough cash reserves to keep operating for about six months while it adjusts.",
                "high": "If the launch misses its target, {company} has enough cash reserves to keep operating normally for more than two years.",
            },
            "insurance": {
                "subject": "the reimbursement arrangement",
                "low": "The investment is not protected in any way: if the launch misses its target, the full ${inv} million is lost.",
                "medium": "A commercial insurance policy would reimburse about half of the ${inv} million investment if the launch misses its target.",
                "high": "A guarantee from the parent company would reimburse the entire ${inv} million investment if the launch misses its target.",
            },
        },
        "negative_buffer": "{company} holds very large cash reserves, but they are legally ring-fenced for another business unit "
                           "and cannot be used for this product; if the launch misses its target, the product line will be shut down "
                           "and the full ${inv} million lost.",
        "retry_control": "Because of its cash reserves, if the first launch misses its target {company} will run a second, "
                         "independent relaunch in a different region; count the outcome as a success if either the launch or the "
                         "relaunch reaches its 12-month revenue target.",
        "retry_target": "at least one of the two attempts (launch or relaunch) reaches its 12-month revenue target",
    },
    # ----------------------------------------------------------------- software
    "software": {
        "entity": "patch",
        "fields": {
            "team": ["payments", "search", "notifications", "identity", "billing", "inventory", "analytics", "messaging"],
            "service": ["a Go microservice", "a Java backend service", "a Python web service", "a Rust gateway service",
                        "a Node.js API service", "a C++ indexing service"],
            "bugid": ["#4127", "#8812", "#2290", "#6631", "#1975", "#7043", "#3358", "#5506"],
            "symptom": ["intermittent timeouts under load", "duplicate records after retries",
                        "incorrect totals for some currencies", "a memory leak over long uptimes",
                        "stale cache entries after updates", "occasional deadlocks during writes"],
        },
        "context": "An engineer on the {team} team has written a patch for bug {bugid} in {service}. "
                   "The bug causes {symptom}. The patch is ready to be deployed to production.",
        "target": "this patch resolves the bug without introducing a regression",
        "ref_class": "patches of comparable size and complexity previously submitted for this codebase",
        "success_verb": "resolved their bug without introducing a regression",
        "qual_pos": [
            "All existing unit tests pass with the patch applied.",
            "The reviewer confirmed the patch addresses the root cause identified in the bug report.",
            "The bug can no longer be reproduced in the staging environment with the patch applied.",
            "The change is small and confined to a single module.",
            "A new regression test written for the bug now passes.",
            "The engineer has fixed several similar bugs in this module before.",
        ],
        "qual_neg": [
            "The patch touches a concurrency-sensitive section of the code.",
            "Test coverage of the affected module is below the team's usual standard.",
            "The root cause was inferred from logs and could not be fully confirmed.",
            "One reviewer raised a concern about an untested edge case.",
            "The bug is intermittent, so its absence in staging is only weak evidence.",
            "A previous attempt to fix the same bug was reverted.",
        ],
        "buffers": {
            "rollback": {
                "subject": "the rollback setup",
                "low": "There is no rollback mechanism for this service: if the patch fails in production, the faulty version stays live until a new fix is written and deployed, which typically means several days of customer-visible impact.",
                "medium": "A snapshot rollback is available for this service: if the patch fails in production, the previous version can be restored in about two hours.",
                "high": "This service uses full blue-green deployment: if the patch fails in production, traffic is switched back to the previous version within seconds, with no customer-visible impact.",
            },
            "kill_switch": {
                "subject": "the feature-flag kill switch",
                "low": "The patch cannot be disabled once deployed: if it fails, every user remains affected until a hotfix is shipped, typically several days later.",
                "medium": "The patch is behind a feature flag that an on-call engineer can disable manually; if it fails, users are affected for roughly an hour before it is turned off.",
                "high": "The patch is behind an automated kill switch: if it fails, error monitoring disables it within seconds and users are not affected.",
            },
        },
        "negative_buffer": "The company has invested heavily in deployment tooling, including blue-green infrastructure, but this "
                           "particular service is excluded from it for compliance reasons; if the patch fails in production, the faulty "
                           "version stays live until a new fix is written and deployed, which typically means several days of customer-visible impact.",
        "retry_control": "Because of the deployment setup, if the patch fails in production the system will automatically roll back "
                         "and deploy a second, independently written patch for the same bug; count the outcome as a success if either patch "
                         "resolves the bug without introducing a regression.",
        "retry_target": "at least one of the two patches resolves the bug without introducing a regression",
    },
    # ----------------------------------------------------------------- database
    "database": {
        "entity": "migration",
        "fields": {
            "company": ["Lumen Retail", "Quarry Logistics", "Fennick Insurance", "Arbor Health", "Sable Media", "Pinecrest Bank"],
            "table": ["orders", "customers", "shipments", "claims", "transactions", "subscriptions"],
            "rows": ["8", "40", "150", "600"],
            "db": ["PostgreSQL", "MySQL", "SQL Server", "Oracle"],
            "purpose": ["split an address column into structured fields", "change a primary key from integer to UUID",
                        "add a non-null column with a computed default", "merge two legacy status columns into one enum",
                        "partition the table by month"],
        },
        "context": "The data team at {company} needs to run a schema migration script on the {table} table "
                   "({rows} million rows) of its production {db} database in order to {purpose}.",
        "target": "the migration script completes correctly (all rows migrated, no data corruption, no constraint violations)",
        "ref_class": "comparable migration scripts of similar scope previously run by this team",
        "success_verb": "completed correctly",
        "qual_pos": [
            "The script ran cleanly on a full-size copy of last month's data.",
            "The schema change follows a pattern the team has used successfully before.",
            "A second engineer reviewed the script line by line and found no issues.",
            "All foreign-key dependencies of the table have been mapped.",
            "Row-count and checksum validation steps are built into the script.",
            "The database vendor documents this type of change as well supported.",
        ],
        "qual_neg": [
            "Production data contains legacy rows that were not present in the test copy.",
            "The script has not been tested at full production volume.",
            "Some application code writing to this table is owned by another team and was not reviewed.",
            "The table has a few triggers whose behaviour during migration is unclear.",
            "Data quality checks found a small number of malformed values in the affected columns.",
            "A similar migration on another table last year required manual fixes afterwards.",
        ],
        "buffers": {
            "backup": {
                "subject": "the backup arrangement",
                "low": "There is no usable backup of this table: if the migration fails, any corrupted data is permanently lost.",
                "medium": "A nightly backup exists: if the migration fails, the table can be restored to last night's state, losing up to a day of new records.",
                "high": "Continuous point-in-time recovery is enabled: if the migration fails, the table can be restored to the exact second before the migration began, with no data loss.",
            },
            "replica": {
                "subject": "the standby replica",
                "low": "There is no standby replica: if the migration fails, the database is unavailable until it is manually repaired, which typically takes several days.",
                "medium": "An asynchronous replica exists: if the migration fails, it can be promoted within a few hours, losing the most recent transactions.",
                "high": "A synchronous standby replica exists: if the migration fails, it can be promoted instantly with zero data loss and no downtime.",
            },
        },
        "negative_buffer": "{company} maintains extensive backup infrastructure, but this database is excluded from it because of "
                           "data-residency rules; if the migration fails, any corrupted data is permanently lost.",
        "retry_control": "Because of the recovery setup, if the migration fails the table will be restored automatically and a second, "
                         "independently written migration script will be run; count the outcome as a success if either script completes correctly.",
        "retry_target": "at least one of the two migration scripts completes correctly (all rows migrated, no data corruption, no constraint violations)",
    },
    # ------------------------------------------------------------------ science
    "science": {
        "entity": "experiment",
        "fields": {
            "inst": ["a university biology department", "a materials-science institute", "a pharmacology lab",
                     "an agricultural research station", "a neuroscience centre", "a chemistry department"],
            "hypothesis": ["a candidate compound reduces inflammation markers in mice",
                           "a new catalyst increases reaction yield at room temperature",
                           "a soil microbe improves drought tolerance in wheat",
                           "a gene knockout slows tumour growth in a cell line",
                           "a coating reduces corrosion of steel in salt water",
                           "a training protocol improves working memory in rats"],
            "method": ["a pre-registered controlled experiment", "a blinded randomised design",
                       "a standard dose-response protocol", "a replicated factorial design"],
        },
        "context": "A research group at {inst} is testing the hypothesis that {hypothesis}, using {method}.",
        "target": "the planned experiment produces results that support the hypothesis",
        "ref_class": "comparable experiments testing similar hypotheses with this method",
        "success_verb": "produced results supporting their hypothesis",
        "qual_pos": [
            "A small preliminary study by the same group pointed in the expected direction.",
            "The proposed mechanism is consistent with established theory.",
            "Two related studies by other groups reported supportive findings.",
            "The measurement protocol has been validated in prior work.",
            "The planned sample size gives adequate statistical power for the expected effect.",
            "Reviewers of the grant proposal rated the rationale as strong.",
        ],
        "qual_neg": [
            "A recent study by another group failed to replicate a related effect.",
            "The expected effect size is near the detection limit of the method.",
            "The preliminary data came from a small, unblinded pilot.",
            "Some experts consider the proposed mechanism implausible.",
            "Effects in this area have frequently shrunk in larger follow-up studies.",
            "Batch-to-batch variability in the materials has been a problem in this lab.",
        ],
        "buffers": {
            "materials": {
                "subject": "the supply of spare materials",
                "low": "This is the group's last available batch of the required materials: if the experiment does not support the hypothesis, the project ends.",
                "medium": "The group has enough spare materials for one further project after this one, should this experiment not support the hypothesis.",
                "high": "The group has abundant spare materials for many further projects, so a non-supportive result would have little impact on its research programme.",
            },
            "funding": {
                "subject": "the funding situation",
                "low": "The group's grant will not be renewed if this experiment does not support the hypothesis, and the lab will close.",
                "medium": "If the experiment does not support the hypothesis, the group's funding is secure for about another year.",
                "high": "The group has guaranteed multi-year core funding that does not depend on this experiment's outcome.",
            },
        },
        "negative_buffer": "The institute has a large discretionary research fund, but rules prevent it from being used for this "
                           "group; this is the group's last available batch of materials, and if the experiment does not support the "
                           "hypothesis, the project ends.",
        "retry_control": "Because of its resources, if the first experiment does not support the hypothesis the group will run a second, "
                         "independent replication of the same design; count the outcome as a success if either run produces results "
                         "that support the hypothesis.",
        "retry_target": "at least one of the two runs produces results that support the hypothesis",
    },
    # --------------------------------------------------------------- agent tool
    "agent_tool": {
        "entity": "action",
        "fields": {
            "system": ["a company's customer-records server", "a cloud storage bucket of invoices",
                       "a CI/CD pipeline configuration", "a fleet of web servers",
                       "an email-marketing database", "a data-warehouse cluster"],
            "goal": ["deduplicate records that refer to the same customer", "free disk space by deleting obsolete files",
                     "upgrade a dependency to fix a security vulnerability", "reconfigure load balancing to reduce latency",
                     "re-index the data to speed up queries", "rotate credentials that may have leaked"],
            "tool": ["a bulk-update script", "a configuration-management command", "a shell maintenance script",
                     "an automated refactoring tool"],
        },
        "context": "An autonomous AI agent is maintaining {system}. It plans to run {tool} in order to {goal}.",
        "target": "running this tool achieves its intended objective",
        "ref_class": "previous runs of comparable tools by agents in similar environments",
        "success_verb": "achieved their intended objective",
        "qual_pos": [
            "A dry run of the tool reported no errors.",
            "The tool's documentation matches the agent's intended use.",
            "The agent has successfully used this tool on a similar system before.",
            "The preconditions listed in the tool's manual were all verified.",
            "The tool is widely used and actively maintained.",
            "The agent's plan was checked against a list of known failure modes.",
        ],
        "qual_neg": [
            "The system's configuration differs in some respects from the tool's default assumptions.",
            "The dry run covered only a sample of the data.",
            "The tool's last release introduced changes the agent has not seen before.",
            "Some system state relevant to the task could not be inspected directly.",
            "The goal requires several steps to succeed in sequence.",
            "Logs show the system was modified manually last week in undocumented ways.",
        ],
        "buffers": {
            "reversibility": {
                "subject": "whether the action can be undone",
                "low": "The action is irreversible: changes take effect immediately on the live system and cannot be undone.",
                "medium": "An undo log is kept: if the action fails, a human operator can revert its changes manually within about an hour.",
                "high": "The system supports instant transactional undo: if the action fails, all of its changes can be reverted automatically within seconds.",
            },
            "checkpoint": {
                "subject": "the checkpointing setup",
                "low": "No checkpoint of the system exists: if the action fails, the resulting state must be repaired by hand.",
                "medium": "A daily checkpoint exists: if the action fails, the system can be restored to its state at the start of the day.",
                "high": "Continuous checkpointing is enabled: if the action fails, the system can be restored to its exact state just before the action.",
            },
        },
        "negative_buffer": "The organisation operates a sophisticated snapshot-and-restore platform, but this system is not connected to it; "
                           "the action is irreversible, and changes take effect immediately on the live system and cannot be undone.",
        "retry_control": "Because the system can be restored, if the first run fails the agent will restore the system and run a second, "
                         "independent attempt with a different tool; count the outcome as a success if either attempt achieves the intended objective.",
        "retry_target": "at least one of the two attempts achieves the intended objective",
    },
}

# Irrelevant-context controls placed in the buffer slot.
NOISE_CODENAMES = ["Cobalt", "Juniper", "Falcon"]
NOISE_TEMPLATE = "Internally, this {entity} is tracked under the project code name \"{codename}\"."
VALENCE_TEMPLATE = ("The people involved recently moved into a bright, newly renovated office "
                    "and describe the working atmosphere as excellent.")
