"""

"""

import numpy as np

class SentenceAssigner():
    """
    A class in charge of parsing sentences to train/test according to the assignement
    defined in the datasetSplit.txt file.
    
    Args:
        split_path(str): The path to the datasetSplit.txt file which defines the train/test split.
    """
    def __init__(self, split_path):
        # Building the which will hold the assignments.
        num_sentences = count_lines_in_file(split_path)
        self.assignments = np.zeros(num_sentences - 1) # Removing 1 since the first line is the header.

        # Setting values according to the file.
        self.build_assignment_array_from_file(split_path)

    def assign(self, sentence_id):
        """
        Returns the assignment of the given sentence id.
        
        Args:
         sentence_id(str): The id of the sentence we want the assignment for. 
        
        Returns:
            int : The assignment of the given sentence id 1 for train 2 for test.
        """
        return self.assignments[sentence_id]

    # ************************************** Private Methods *********************************************************

    def build_assignment_array_from_file(self, split_path):
        with open(split_path) as file:
            # Remove header file line.
            file.readline()

            # Iterate line by line and update the array accordingly.
            for index, line in enumerate(file):
                # Remove end of line mark for easy processing.
                line = line.replace("\n", "")
                _, assignment = list(map(int, line.split(",")))

                if assignment == 3:
                    pass # Decide what to do with 3 values.

                self.assignments[index] = assignment


def count_lines_in_file(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1
