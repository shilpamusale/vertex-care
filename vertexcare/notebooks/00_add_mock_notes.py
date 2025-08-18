import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Configure basic logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_mock_notes():
    """
    Generates a list of diverse, realistic mock CHW notes.
    """
    notes = [
        # --- Type 1: Clear & Actionable ---
        "Patient reports running out of metformin. "
        + "A 30-day refill was called into CVS on Main St. "
        + "Confirmed patient has transportation for "
        + "pickup on Friday, 8/19. Scheduled follow-up call for 8/23.",
        "Delivered groceries and confirmed patient has enough food "
        + "for the week. Blood pressure reading was 130/85. ",
        "Patient understands medication schedule. "
        + "Transportation for cardiology appointment on "
        + "8/22 confirmed with medical taxi service."
        + "Patient has appointment card.",
        # --- Type 2: Long, Narrative Note ---
        "Had a long conversation with the patient today."
        + "They seemed to be in good spirits after seeing their grandkids."
        + "They mentioned that their breathing feels a bit tight in the "
        + " mornings but it gets better after they use their"
        + " albuterol inhaler. They are still worried about "
        + "the cost of their other medications, so I spent some "
        + "time explaining the new insurance plan and provided "
        + "a number for the patient advocate service. "
        + "They still haven't scheduled "
        + "their follow-up with Dr. Smith, "
        + "citing transportation issues again.",
        # --- Type 3: Contradictory or Evolving Note ---
        "Patient initially stated they had a ride to their PCP appointment."
        + "Later in the call, they admitted their son is no "
        + "longer available and they have no other transportation. "
        + "Original plan is no longer viable. "
        + "ACTION: Arrange for a medical taxi. "
        + "NOTE: Patient refused the medical taxi, "
        + "citing privacy concerns. "
        + "New plan is to reschedule the appointment for next "
        + "week when daughter can drive.",
        +"Initially reported taking all medications correctly. "
        + "Upon review of pillbox, "
        + "it was clear the morning dose of lisinopril "
        + "was missed for the past two days. "
        + "Re-educated patient on schedule "
        + "and set up a daily reminder call.",
        # --- Type 4: Vague or Ambiguous Note ---
        "Patient seems down today. Complained of feeling 'off'. "
        + "Mentioned not eating much. Might need "
        + "someone to check in on them soon.",
        +"Subject was quiet during the visit. "
        + "Didn't say much about their health, "
        + "just that 'things are the same'.",
        # --- Type 5: Jargon-Heavy / Abbreviated Note ---
        "Pt c/o SOB. BP 140/90. Rx for lisinopril needs refill. "
        + "F/u w/ PCP scheduled 8/25. "
        + "Referred to SNAP for food insecurity.",
        +"NPO after midnight for procedure tomorrow. "
        + "Pt verbalized understanding of instructions."
        + "All pre-op questions answered.",
    ]
    return notes


def add_notes_to_dataset(
    input_path: Path, output_path: Path, fill_fraction: float = 0.3
):
    """
    Reads a CSV, adds a 'chw_notes' column, and populates a fraction of
    the rows with mock notes.

    Args:
        input_path: Path to the original CSV file.
        output_path: Path to save the new CSV file.
        fill_fraction: The fraction of
                       rows to populate with notes
                       (e.g., 0.3 for 30%).
    """
    if not input_path.exists():
        logging.error(f"Input file not found at: {input_path}")
        raise FileNotFoundError(f"Input file not found at: {input_path}")

    logging.info(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path)

    logging.info("Creating mock notes...")
    mock_notes = create_mock_notes()

    # Initialize the new column with empty strings
    df["chw_notes"] = ""

    # Get a random sample of indices to fill with notes
    num = int(len(df) * fill_fraction)
    random_indices = np.random.choice(df.index, size=num, replace=False)

    # Assign random notes to the selected indices
    notes_to_assign = np.random.choice(mock_notes, size=num, replace=True)
    df.loc[random_indices, "chw_notes"] = notes_to_assign

    logging.info(f"Added notes to {num} out of {len(df)} rows.")

    # Save the updated dataframe
    logging.info(f"Saving updated dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    logging.info("Script finished successfully.")


if __name__ == "__main__":
    # Assuming the script is run from the root of the vertexcare project
    project_root = Path(__file__).resolve().parent

    # If you place this script in /notebooks, the root is one level up
    if project_root.name == "notebooks":
        project_root = project_root.parent

    raw_data_dir = project_root / "data" / "01_raw"

    input_csv = raw_data_dir / "mock_patient_readmission_data.csv"
    output_csv = raw_data_dir / "mock_data_with_notes.csv"

    add_notes_to_dataset(input_path=input_csv, output_path=output_csv)
