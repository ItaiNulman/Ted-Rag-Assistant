import pandas as pd

def inspect_data():
    print("--- Loading Data ---")
    try:
        # load the data
        df = pd.read_csv('ted_talks_en.csv')
        print(f"Data loaded successfully! Total rows: {len(df)}")
        
        # print the columns 
        print("\n--- Columns ---")
        print(df.columns.tolist())
        
        # print a sample row to understand the structure
        print("\n--- Sample Row (First Talk) ---")
        first_row = df.iloc[0]
        print(f"Title: {first_row['title']}")
        print(f"Speaker: {first_row['speaker_1']}")
        # print the start of the transcript to avoid cluttering the screen
        print(f"Transcript Start: {str(first_row['transcript'])[:200]}...")
        
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

if __name__ == "__main__":
    df = inspect_data()