import requests
import pandas as pd
import time

def fetch_trials(condition="cancer", max_pages=5):
    """Pull real clinical trial data from ClinicalTrials.gov"""
    
    url = "https://clinicaltrials.gov/api/v2/studies"
    all_studies = []
    next_token = None
    page = 1

    print(f"Fetching real clinical trials for: {condition}")
    print("-" * 50)

    while page <= max_pages:
        params = {
            "query.cond": condition,
            "fields": (
                "NCTId,BriefTitle,OverallStatus,Phase,"
                "EnrollmentCount,StartDate,LeadSponsorName,"
                "EligibilityCriteria,StudyType"
            ),
            "pageSize": 100,
            "format": "json"
        }
        if next_token:
            params["pageToken"] = next_token

        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        studies = data.get("studies", [])
        all_studies.extend(studies)
        
        print(f"  Page {page}: pulled {len(studies)} trials (total: {len(all_studies)})")
        
        next_token = data.get("nextPageToken")
        if not next_token:
            break
        
        page += 1
        time.sleep(0.5)  # be polite to the API

    print(f"\nTotal studies fetched: {len(all_studies)}")
    return all_studies

def parse_to_dataframe(studies):
    """Flatten the nested JSON into a clean flat table"""
    records = []
    
    for s in studies:
        proto = s.get("protocolSection", {})
        
        # Pull from nested sections
        id_module       = proto.get("identificationModule", {})
        status_module   = proto.get("statusModule", {})
        design_module   = proto.get("designModule", {})
        sponsor_module  = proto.get("sponsorCollaboratorsModule", {})
        eligibility_mod = proto.get("eligibilityModule", {})

        # Eligibility criteria text — count words as proxy for complexity
        elig_text = eligibility_mod.get("eligibilityCriteria", "")
        elig_word_count = len(elig_text.split()) if elig_text else 0

        records.append({
            "trial_id":          id_module.get("nctId"),
            "title":             id_module.get("briefTitle"),
            "status":            status_module.get("overallStatus"),
            "phase":             ", ".join(design_module.get("phases", [])),
            "enrollment":        design_module.get("enrollmentInfo", {}).get("count"),
            "enrollment_type":   design_module.get("enrollmentInfo", {}).get("type"),
            "study_type":        design_module.get("studyType"),
            "sponsor":           sponsor_module.get("leadSponsor", {}).get("name"),
            "sponsor_class":     sponsor_module.get("leadSponsor", {}).get("class"),
            "eligibility_words": elig_word_count,
            "start_date":        status_module.get("startDateStruct", {}).get("date"),
            "completion_date":   status_module.get("primaryCompletionDateStruct", {}).get("date"),
        })
    
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Fetch real data
    studies = fetch_trials(condition="cancer", max_pages=5)
    
    # Parse into DataFrame
    df = parse_to_dataframe(studies)
    
    # Save to CSV
    output_path = "~/ctios-ml/real_trials.csv"
    df.to_csv(output_path, index=False)
    
    print(f"\n=== DATASET SUMMARY ===")
    print(f"Shape: {df.shape[0]} trials, {df.shape[1]} columns")
    print(f"\nColumn names:\n  {list(df.columns)}")
    print(f"\nStatus breakdown:")
    print(df["status"].value_counts().head(10).to_string())
    print(f"\nPhase breakdown:")
    print(df["phase"].value_counts().head(10).to_string())
    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())
    print(f"\n✅ Saved to {output_path}")
