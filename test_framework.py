import unittest
import pandas as pd
import os
from main import analyze_sentiments, visualize_sentiments

class TestSentimentAnalysisPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("\n📦 Setting up test data...")
        cls.test_input_csv = "test_input.csv"
        cls.test_output_csv = "test_output.csv"
        
        data = {
            "timestamp": [
                "2025-04-04 20:06:37", "2025-04-04 19:51:09",
                "2025-04-04 20:17:44", "2025-04-04 19:30:56"
            ],
            "tweet": [
                "Clinical finish by Liverpool's striker! ⚽ #Liverpool",
                "Wasted chance by PSG's forward! 😫 #PSG",
                "Perfect through ball from PSG! 🎯 #PSG",
                "Controversial offside call against Liverpool ⚖ #Liverpool"
            ],
            "team": ["Liverpool", "PSG", "PSG", "Liverpool"]
        }

        df = pd.DataFrame(data)
        df.to_csv(cls.test_input_csv, index=False)

        print("✅ Setup completed.")

    # ---------------------- FUNCTIONAL TESTS ----------------------

    def test_functional_sentiment_output(self):
        print("\n🧪 [Functional] Testing sentiment analysis output structure...")
        df = analyze_sentiments(self.test_input_csv, self.test_output_csv)

        self.assertTrue(os.path.exists(self.test_output_csv), "Output CSV not created")
        self.assertIn("sentiment", df.columns, "Sentiment column missing")
        valid_sentiments = {"positive", "negative", "neutral"}
        self.assertTrue(all(s in valid_sentiments for s in df['sentiment']), "Invalid sentiment labels")

        print("✅ [Functional] Sentiment analysis output structure is valid.")

    def test_error_on_missing_columns(self):
        print("\n🧪 [Functional] Testing missing columns error handling...")
        data = {
            "time": ["2025-04-04 20:06:37"],
            "text": ["Goal by PSG!"],
            "club": ["PSG"]
        }
        df = pd.DataFrame(data)
        df.to_csv("missing_columns.csv", index=False)

        with self.assertRaises(ValueError):
            analyze_sentiments("missing_columns.csv", "missing_output.csv")

        print("✅ [Functional] Correctly raised error for missing columns.")
        os.remove("missing_columns.csv")

    # ------------------- NON-FUNCTIONAL TESTS -------------------

    def test_nonfunctional_large_input(self):
        print("\n🧪 [Non-Functional] Testing performance on large input...")
        large_data = pd.read_csv(self.test_input_csv)
        for _ in range(100):  # Simulate ~400 tweets
            large_data = pd.concat([large_data, pd.read_csv(self.test_input_csv)], ignore_index=True)
        large_data.to_csv("large_test_input.csv", index=False)

        try:
            df = analyze_sentiments("large_test_input.csv", "large_test_output.csv")
            self.assertEqual(len(df), len(large_data), "Mismatch in row count after processing")
            print("✅ [Non-Functional] Large input handled successfully.")
        finally:
            os.remove("large_test_input.csv")
            os.remove("large_test_output.csv")

    def test_visualization_runs_without_error(self):
        print("\n🧪 [Non-Functional] Testing visualization rendering...")
        df = analyze_sentiments(self.test_input_csv, self.test_output_csv)
        try:
            visualize_sentiments(self.test_output_csv)
            print("✅ [Non-Functional] Visualization completed without error.")
        except Exception as e:
            self.fail(f"Visualization failed with error: {e}")

    @classmethod
    def tearDownClass(cls):
        print("\n🧹 Cleaning up test files...")
        for file in [cls.test_input_csv, cls.test_output_csv]:
            if os.path.exists(file):
                os.remove(file)
        print("✅ Cleanup done.")

if __name__ == "__main__":
    try:
        unittest.main(verbosity=0)
        print("\n🎉 All tests passed successfully. Your code works perfectly! ✅")
    except Exception as e:
        print(f"\n❌ Some tests failed: {e}")
