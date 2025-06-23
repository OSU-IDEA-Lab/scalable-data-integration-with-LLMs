import copy


def is_better_match(score1, score2):
    if score1 is None or score2 is None:
        return False
    return score1 > score2

def lowercase_keys_and_values(data):
    result = {}
    for key, value in data.items():
        lower_key = key.lower()
        if isinstance(value, dict):
            lower_value = [
                [attr.lower(), score]
                for attr, score in value.items()
            ]
        else:
            lower_value = [
                [item[0].lower()] + item[1:] if isinstance(item[0], str) else item
                for item in value
            ]
        result[lower_key] = lower_value
    return result


class ManyToManyStableMatcher:
    def __init__(self, schema_a, schema_b, conf_a, conf_b, top_k=10, is_logits=False):
        self.schema_a = [x.lower() for x in schema_a]  # List of attributes in Schema A
        self.schema_b = [x.lower() for x in schema_b]  # List of attributes in Schema B
        self.conf_a = lowercase_keys_and_values(
            conf_a)  # Dictionary with Schema A attributes as keys and their sorted preferences as values
        self.conf_b = lowercase_keys_and_values(
            conf_b)  # Dictionary with Schema B attributes as keys and their sorted preferences as values
        self.top_k = top_k  # Limit on number of matches per attribute, if any
        self.pref_a = {}
        self.pref_b = {}
        self.pref_a_temp = {}
        self.pref_b_temp = {}
        if not is_logits:
            self.no_match = "none of the options"
        else:
            self.no_match = "there is no match."



    def find_score_in_preferences(self, preferences, attribute):

        for idx, (attr, score) in enumerate(preferences):
            if attr == attribute:
                return score
        return None

    def match(self):
        """Perform the matching process and return the stable matches set M, along with round-wise results."""
        M = []  # Stable match set

        acceptable_match_a = {a: [] for a in self.schema_a}  # Matches acceptable to attributes in Schema A
        acceptable_match_b = {b: [] for b in self.schema_b}  # Matches acceptable to attributes in Schema B
        # acceptable_match.update({b: [] for b in self.schema_b})  # Matches acceptable to attributes in Schema B

        for a in self.schema_a:
            self.pref_a[a] = [[b, score] for b, score in self.conf_a[a] if b != self.no_match]
            acceptable_match_a[a] = [b for b, _ in self.pref_a[a]]

        for b in self.schema_b:
            # print(self.conf_b)
            self.pref_b[b] = [[a, score] for a, score in self.conf_b[b] if a != self.no_match]
            acceptable_match_b[b] = [a for a, _ in self.pref_b[b]]

        self.pref_a_temp = copy.deepcopy(self.pref_a)
        self.pref_b_temp = copy.deepcopy(self.pref_b)
        # self.pref_b_temp = self.pref_b.copy()

        # print('self.schema_a')
        # # print(self.schema_a)
        # print(self.pref_a)
        # print('self.schema_b')
        # # print(self.schema_b)
        # print(self.pref_b)

        r = 1  # Round counter
        rounds = []  # To track round-wise results > top-k

        # Repeat matching process until max rounds or no new matches
        while r <= self.top_k:
            # print(f"--- round {r}")
            match_made = False  # Flag to track if any new match is made in the current round

            free_a = {a: True for a in self.schema_a}  # Free attributes in Schema A
            free_b = {b: True for b in self.schema_b}  # Free attributes in Schema B

            # print("M: ",M)
            # print("r: ", r)
            # print('acceptable_match_a')
            # print(acceptable_match_a)
            # print('acceptable_match_b')
            # print(acceptable_match_b)
            # print('pref_a')
            # print(self.pref_a)
            # print('pref_b')
            # print(self.pref_b)

            while any(free_a[a] and len(self.pref_a[a]) > 0 for a in
                      self.schema_a):  # While there are free attributes in Schema A
                # print("r: ",r)

                # print('free_a')
                # print(free_a)
                # print('free_b')
                # print(free_b)
                for a in list(free_a.keys()):
                    # print("M: ", M)
                    # print("\n\nfree_a[a]: ", a, free_a[a])
                    # print("free_a: ", free_a)
                    # print("free_b: ", free_b)
                    if not free_a[a]:  # Skip if no preferences are left
                        continue

                    if len(self.pref_a[a]) == 0:
                        continue

                    # print("pref_a[a]: ", self.pref_a[a])

                    b = self.pref_a[a].pop(0)[0]  # A proposes to its most preferred B
                    # print("b: ", b)
                    # print("r: ", r)
                    # print("a: ", a)
                    # print("b: ", b)
                    # if b in acceptable_match_b:
                    # print("acceptable_match_b: ", acceptable_match_b[b])
                    # print("free_b[b]: ", free_b[b])
                    if b in acceptable_match_b and a in acceptable_match_b[b]:
                        # print(" a in acceptable_match[b]")
                        if b in free_b and free_b[b]:
                            M.append((a, b))
                            match_made = True
                            # print('match made ::::::::::::::::::::::::::::::: (a, b)')
                            # print((a, b))
                            free_b[b] = False
                            free_a[a] = False
                            # print("MATCHED")
                        else:
                            # If B is already matched, check preferences
                            current_match = next(((a1, b1) for a1, b1 in M if b1 == b), None)
                            # print("current match: ", current_match)
                            # print(self.pref_b[b])
                            if current_match and is_better_match(self.find_score_in_preferences(self.pref_b[b], a),
                                                                 self.find_score_in_preferences(self.pref_b[b],
                                                                                                current_match[0])):
                                a2 = current_match[0]
                                # print("a2: ", a2)
                                # print("MATCHED")
                                M.remove((a2, b))  # Remove the worst match
                                M.append((a, b))  # Replace with the new match
                                match_made = True
                                # print('match made ::::::::::::::::::::::::::::::: (a, b)')
                                # print((a, b))
                                free_b[b] = False
                                free_a[a] = False
                                free_a[a2] = True  # Re-add the previous match to free list

            # print("r: ", r)
            # print("M: ", M)
            # print('acceptable_match_a')
            # print(acceptable_match_a)
            # print('acceptable_match_b')
            # print(acceptable_match_b)
            # print('pref_a')
            # print(self.pref_a)
            # print('pref_b')
            # print(self.pref_b)
            for match in M:
                a_ = match[0]
                b_ = match[1]
                if b_ in acceptable_match_a[a_]:
                    acceptable_match_a[a_].remove(b_)
                if a_ in acceptable_match_b[b_]:
                    acceptable_match_b[b_].remove(a_)
                # print("\n\na ", a_, "b ", b_)
                # print("self.pref_b[b_] ", self.pref_b[b_])
                # for pair in self.pref_b[b_]:
                #     print("pair ", pair)
                self.pref_b_temp[b_] = [pair for pair in self.pref_b_temp[b_] if pair[0] != a_]

                # print("self.pref_b[b_] ", self.pref_b[b_])
                # print()
                # print("self.pref_a[a_] ", self.pref_a[a_])
                # for pair in self.pref_b[b_]:
                # print("pair ", pair)
                self.pref_a_temp[a_] = [pair for pair in self.pref_a_temp[a_] if pair[0] != b_]
                # print("self.pref_a[a_] ", self.pref_a[a_])

            # self.pref_b = self.pref_b_temp.copy()
            # self.pref_a = self.pref_a_temp.copy()

            self.pref_b = copy.deepcopy(self.pref_b_temp)
            self.pref_a = copy.deepcopy(self.pref_a_temp)
            # print("M: ",M)
            # print("r: ", r)
            # print('acceptable_match_a')
            # print(acceptable_match_a)
            # print('acceptable_match_b')
            # print(acceptable_match_b)
            # print('pref_a')
            # print(self.pref_a)
            # print('pref_b')
            # print(self.pref_b)
            # print('free_a')
            # print(free_a)
            #
            # print("....................")

            # exit(0)
            if not match_made:
                # print("force break ", r)
                break  # Exit if no new matches were formed in this round

            rounds.append(copy.deepcopy(M))  # k-th element is top-k match
            r += 1  # Increment round counter

        return M, rounds
