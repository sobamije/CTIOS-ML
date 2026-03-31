
import requests
import pandas as pd
import time

def fetch_trials(condition="cancer", max_pages=20):
    url = "https://clinicaltrials.gov/api/v2/studies"
    all_studies = []
    next_token = None
    page = 1
    print(f"Fetching trials for: {condition}")
    print("-" * 50)
    while page <= max_pages:
        params = {
            "query.cond": condition,
            "fields": "NCTId,BriefTitle,OverallStatus,Phase,EnrollmentCount,StartDate,LeadSponsorName,EligibilityCriteria,StudyType,Condition,LocationCountry",
            "pageSize": 100,
            "format": "json"
        }
        if next_token:
            params["pageToken"] = next_token
        response = requests.get(url, params=params, timeout=30)
        if response.status_code != 200:
            print(f"  Page {page}: error {response.status_code}, stopping")
            break
        data = response.json()
        studies = data.get("studies", [])
        all_studies.extend(studies)
        print(f"  Page {page}: pulled {len(studies)} trials (total: {len(all_studies)})")
        next_token = data.get("nextPageToken")
        if not next_token:
            break
        page += 1
        time.sleep(0.5)
    print(f"Total fetched: {len(all_studies)}")
    return all_studies

def parse_to_dataframe(studies):
    records = []
    for s in studies:
        proto = s.get("protocolSection", {})
        id_mod     = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design_mod = proto.get("designModule", {})
        sponsor_mod= proto.get("sponsorCollaboratorsModule", {})
        elig_mod   = proto.get("eligibilityModule", {})
        conds_mod  = proto.get("conditionsModule", {})
        contacts   = proto.get("contactsLocationsModule", {})
        elig_text  = elig_mod.get("eligibilityCriteria", "")
        elig_words = len(elig_text.split()) if elig_text else 0
        conditions = conds_mod.get("conditions", [])
        primary    = conditions[0].lower() if conditions else ""
        locations  = contacts.get("locations", [])
        site_count = len(locations)
        countries  = list(set(l.get("country","") for l in locations if l.get("country")))
        records.append({
            "trial_id":          id_mod.get("nctId"),
            "title":             id_mod.get("briefTitle"),
            "status":            status_mod.get("overallStatus"),
            "phase":             ", ".join(design_mod.get("phases", [])),
            "enrollment":        design_mod.get("enrollmentInfo", {}).get("count"),
            "enrollment_type":   design_mod.get("enrollmentInfo", {}).get("type"),
            "study_type":        design_mod.get("studyType"),
            "sponsor":           sponsor_mod.get("leadSponsor", {}).get("name", ""),
            "sponsor_class":     sponsor_mod.get("leadSponsor", {}).get("class", ""),
            "eligibility_words": elig_words,
            "start_date":        status_mod.get("startDateStruct", {}).get("date"),
            "completion_date":   status_mod.get("primaryCompletionDateStruct", {}).get("date"),
            "primary_condition": conditions[0] if conditions else "",
            "site_count":        site_count,
            "country_count":     len(countries),
            "is_multinational":  int(len(countries) > 1),
            "is_oncology":       int(any(w in primary for w in ["cancer","tumor","leukemia","lymphoma","carcinoma","melanoma","sarcoma"])),
            "is_rare":           int(any(w in primary for w in ["rare","orphan","syndrome","disorder","deficiency"])),
            "is_neuro":          int(any(w in primary for w in ["neuro","alzheimer","parkinson","epilepsy","seizure","brain"])),
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    studies = fetch_trials()
    df = parse_to_dataframe(studies)
    df.to_csv("/Users/sophiaobamije/ctios-ml/real_trials.csv", index=False)
    print(f"Shape: {df.shape}")
    print(f"Avg site count: {df['site_count'].mean():.1f}")
    print(df["status"].value_counts().head(6).to_string())
    print("Saved.")
