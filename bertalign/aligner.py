import re
from typing import List, Optional

import numpy as np

from bertalign import model
from bertalign.corelib import *
from bertalign.utils import clean_text, detect_lang, split_sents, LANG


class Bertalign:
    """
    Performs sentence alignment between two texts or lists of sentences.
    """

    def __init__(self,
                 src: Optional[str] = None,
                 tgt: Optional[str] = None,
                 src_sents_list: Optional[List[str]] = None,
                 tgt_sents_list: Optional[List[str]] = None,
                 max_align: int = 5,
                 top_k: int = 3,
                 win: int = 5,
                 skip: float = -0.1,
                 margin: bool = True,
                 len_penalty: bool = True,
                 is_split: bool = False,
                 clean_input_lists: bool = True,
                 verbose: bool = False # Added verbose flag to control printing
                 ):
        """
        Initializes the Bertalign object.

        Args:
            src (Optional[str]): The source text as a single string.
            tgt (Optional[str]): The target text as a single string.
            src_sents_list (Optional[List[str]]): The source text as a list of sentences.
            tgt_sents_list (Optional[List[str]]): The target text as a list of sentences.
            max_align (int): Maximum number of sentences in a single alignment block (N+M <= max_align).
            top_k (int): Number of nearest neighbors for the first pass.
            win (int): Window size for the second pass.
            skip (float): Penalty for skipping (insertion/deletion).
            margin (bool): Whether to use modified cosine similarity (considering neighbors).
            len_penalty (bool): Whether to apply a penalty for length difference.
            is_split (bool): If True and input are src/tgt strings, use `splitlines` instead of `split_sents`.
            clean_input_lists (bool): If True and input are sentence lists, apply basic cleaning (strip, normalize space) to each sentence.
            verbose (bool): If True, print progress messages. Defaults to False.
        """
        self.max_align = max_align
        self.top_k = top_k
        self.win = win
        self.skip = skip
        self.margin = margin
        self.len_penalty = len_penalty
        self.verbose = verbose # Store verbose flag

        src_sents: List[str] = []
        tgt_sents: List[str] = []
        src_lang_code: str = 'unknown'
        tgt_lang_code: str = 'unknown'

        if src_sents_list is not None and tgt_sents_list is not None:
            if self.verbose: print("[Bertalign] Using pre-split sentence lists as input.")
            src_sents = src_sents_list
            tgt_sents = tgt_sents_list

            if clean_input_lists:
                if self.verbose: print("[Bertalign] Cleaning provided sentence lists...")
                cleaned_src = []
                for line in src_sents:
                    line = line.strip()
                    if line:
                        line = re.sub(r'\s+', ' ', line)
                        cleaned_src.append(line)
                src_sents = cleaned_src if cleaned_src else [''] # Ensure list is not empty if all lines were blank
                cleaned_tgt = []
                for line in tgt_sents:
                    line = line.strip()
                    if line:
                        line = re.sub(r'\s+', ' ', line)
                        cleaned_tgt.append(line)
                tgt_sents = cleaned_tgt if cleaned_tgt else [''] # Ensure list is not empty if all lines were blank
                if self.verbose:
                    print(
                        f"[Bertalign] Cleaning finished. Sentences after cleaning: src={len(src_sents)}, tgt={len(tgt_sents)}")

            # Attempt language detection on the potentially cleaned lists
            try:
                # Detect only if list is not empty and first element is not empty string
                if src_sents and src_sents[0]:
                    src_lang_code = detect_lang(" ".join(src_sents[:5]))
                if tgt_sents and tgt_sents[0]:
                    tgt_lang_code = detect_lang(" ".join(tgt_sents[:5]))
            except Exception as lang_detect_err:
                print(f"[Bertalign] Warning: Language detection failed on sentence lists: {lang_detect_err}")
                # Keep lang codes as 'unknown'

        elif src is not None and tgt is not None:
            if self.verbose: print("[Bertalign] Using string input, performing internal cleaning and splitting.")
            src_input_str = src
            tgt_input_str = tgt
            src_input_str = clean_text(src_input_str)
            tgt_input_str = clean_text(tgt_input_str)
            try:
                src_lang_code = detect_lang(src_input_str)
                tgt_lang_code = detect_lang(tgt_input_str)
            except Exception as lang_detect_err:
                print(f"[Bertalign] Warning: Language detection failed on string input: {lang_detect_err}")
                # Keep lang codes as 'unknown'

            if is_split:
                if self.verbose: print(f"[Bertalign] Splitting input strings using splitlines() (is_split=True)")
                src_sents = src_input_str.splitlines()
                tgt_sents = tgt_input_str.splitlines()
                # Filter out empty lines after split
                src_sents = [s for s in src_sents if s.strip()]
                tgt_sents = [s for s in tgt_sents if s.strip()]
            else:
                if self.verbose:
                    print(
                        f"[Bertalign] Splitting input strings using sentence_splitter for {src_lang_code}/{tgt_lang_code}")

                try:
                    src_sents = split_sents(src_input_str, src_lang_code)
                except Exception as e:
                    print(
                        f"[Bertalign] ERROR splitting source text with sentence_splitter: {e}. Falling back to splitlines.")
                    src_sents = src_input_str.splitlines()
                    src_sents = [s for s in src_sents if s.strip()] # Filter empty lines
                try:
                    tgt_sents = split_sents(tgt_input_str, tgt_lang_code)
                except Exception as e:
                    print(
                        f"[Bertalign] ERROR splitting target text with sentence_splitter: {e}. Falling back to splitlines.")
                    tgt_sents = tgt_input_str.splitlines()
                    tgt_sents = [s for s in tgt_sents if s.strip()] # Filter empty lines


        else:
            raise ValueError(
                "Bertalign requires either (src, tgt) string arguments or (src_sents_list, tgt_sents_list) list arguments.")

        # Ensure sentence lists are never truly empty for downstream processing, use a list with one empty string if needed.
        if not src_sents: src_sents = ['']
        if not tgt_sents: tgt_sents = ['']

        self.src_num = len(src_sents)
        self.tgt_num = len(tgt_sents)

        # Convert detected lang code (like 'en') to expected format (like 'ENG')
        self.src_lang = LANG.ISO.get(src_lang_code, src_lang_code.upper())
        self.tgt_lang = LANG.ISO.get(tgt_lang_code, tgt_lang_code.upper())

        if self.verbose:
            print(f"Source language: {self.src_lang}, Number of sentences: {self.src_num}")
            print(f"Target language: {self.tgt_lang}, Number of sentences: {self.tgt_num}")

        if self.verbose: print(f"Embedding source and target text using {model.model_name} ...")
        try:
            # Pass max_align-1 as per original logic for vec/len calculation
            self.src_vecs, self.src_lens = model.transform(src_sents, self.max_align - 1)
            self.tgt_vecs, self.tgt_lens = model.transform(tgt_sents, self.max_align - 1)
        except Exception as embed_err:
            # This is a critical error, so print it regardless of verbosity
            print(f"[Bertalign] ERROR during sentence embedding: {embed_err}")
            # Re-raise as a runtime error for clarity
            raise RuntimeError(f"Failed to embed sentences: {embed_err}") from embed_err

        # Calculate character ratio based on the lengths of the *first* segments (0-th index)
        sum_tgt_lens_0 = np.sum(self.tgt_lens[0,]) # Sum lengths of first N sentences up to max_align-1
        self.char_ratio = np.sum(self.src_lens[0,]) / sum_tgt_lens_0 if sum_tgt_lens_0 > 0 else 1.0

        self.src_sents = src_sents
        self.tgt_sents = tgt_sents

    def align_sents(self):
        """
        Performs two-pass dynamic programming for sentence alignment.
        The result is stored in self.result.
        """

        # Check for empty inputs before proceeding
        if self.src_num == 0 or self.tgt_num == 0 or (self.src_num == 1 and not self.src_sents[0]) or (
                self.tgt_num == 1 and not self.tgt_sents[0]):
            print("[Bertalign] Warning: Empty sentence list(s) provided. Skipping alignment.")
            self.result = []
            return

        if self.verbose: print("Performing first-step alignment ...")
        # Find top k similar target sentences for each source sentence
        D, I = find_top_k_sents(self.src_vecs[0, :], self.tgt_vecs[0, :], k=self.top_k)
        # Define alignment types for the first pass (likely simpler, e.g., 1:1, 1:0, 0:1)
        first_alignment_types = get_alignment_types(2) # Assuming max 1-1, 1-0, 0-1 in first pass
        # Initialize weights and path matrix for dynamic programming
        first_w, first_path = find_first_search_path(self.src_num, self.tgt_num)
        # Execute the first pass alignment algorithm
        first_pointers = first_pass_align(self.src_num, self.tgt_num, first_w, first_path, first_alignment_types, D, I)
        # Backtrack to find the best alignment path from the first pass
        first_alignment = first_back_track(self.src_num, self.tgt_num, first_pointers, first_path,
                                           first_alignment_types)

        if self.verbose: print("Performing second-step alignment ...")
        # Define alignment types for the second pass (up to max_align)
        second_alignment_types = get_alignment_types(self.max_align)
        # Determine the search space (window) for the second pass based on the first alignment
        second_w, second_path = find_second_search_path(first_alignment, self.win, self.src_num, self.tgt_num)
        # Execute the second pass alignment using full embeddings, lengths, and penalties
        second_pointers = second_pass_align(self.src_vecs, self.tgt_vecs, self.src_lens, self.tgt_lens,
                                            second_w, second_path, second_alignment_types,
                                            self.char_ratio, self.skip, margin=self.margin,
                                            len_penalty=self.len_penalty)
        # Backtrack to find the final, refined alignment path
        second_alignment = second_back_track(self.src_num, self.tgt_num, second_pointers, second_path,
                                             second_alignment_types)

        if self.verbose:
            print(
                f"Finished! Successfully aligning {self.src_num} {self.src_lang} sentences to {self.tgt_num} {self.tgt_lang} sentences\n")
        self.result = second_alignment

    def print_sents(self):
        """Prints the aligned sentence pairs based on the alignment result."""
        if not hasattr(self, 'result') or self.result is None:
            print("[Bertalign] No alignment result available to print.")
            return
        for bead in (self.result):
            # bead[0] contains indices for source, bead[1] for target
            src_line = self._get_line(bead[0], self.src_sents)
            tgt_line = self._get_line(bead[1], self.tgt_sents)
            print(src_line + "\n" + tgt_line + "\n")

    @staticmethod
    def _get_line(bead: List[int], lines: List[str]) -> str:
        """
        Helper function to join sentences based on alignment indices (bead).
        Args:
            bead (List[int]): A list of sentence indices (e.g., [2, 3] for sentences 2 and 3).
            lines (List[str]): The list of all sentences (source or target).
        Returns:
            str: The joined sentences corresponding to the bead indices, or an empty string if bead is empty or invalid.
        """
        line = ''
        if bead and lines: # Check if bead and lines are not empty/None
            start_idx = bead[0]
            end_idx = bead[-1] + 1 # Slice index is exclusive
            # Basic validation of indices
            if 0 <= start_idx < len(lines) and 0 < end_idx <= len(lines) and start_idx < end_idx:
                line = ' '.join(lines[start_idx:end_idx])
            else:
                # Print a warning for invalid indices, useful for debugging
                print(f"[Bertalign Warning] Invalid bead indices for _get_line: {bead} with line count {len(lines)}")
        return line
