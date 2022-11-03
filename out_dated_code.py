
def old_find_repeating_variables(self):
    # TODO fix repeated_variable list from showing forward slashes
    repeating_variables = []
    # TODO Make this more efficient
    if self.dimensional_matrix.rank == 1:
        for parameter1 in self.parameters[1:]:
            repeating_variables.append(parameter1)
    if self.dimensional_matrix.rank == 2:
        for parameter1 in self.parameters[1:]:
            for parameter2 in self.parameters[1:]:
                group2 = ListOfParameters([parameter1, parameter2])
                M = DimensionalMatrix(group2)
                if M.rank == self.dimensional_matrix.rank:
                    if not group2.included_within(repeating_variables):
                        repeating_variables.append(group2)
    if self.dimensional_matrix.rank == 3:
        for parameter1 in self.parameters[1:]:
            for parameter2 in self.parameters[1:]:
                for parameter3 in self.parameters[1:]:
                    group3 = ListOfParameters([parameter1, parameter2, parameter3])
                    M = DimensionalMatrix(group3)
                    if M.rank == self.dimensional_matrix.rank:
                        if not group3.included_within(repeating_variables):
                            repeating_variables.append(group3)
    if self.dimensional_matrix.rank == 4:
        for parameter1 in self.parameters[1:]:
            for parameter2 in self.parameters[1:]:
                for parameter3 in self.parameters[1:]:
                    for parameter4 in self.parameters[1:]:
                        group4 = ListOfParameters([parameter1, parameter2, parameter3, parameter4])
                        M = DimensionalMatrix(group4)
                        if M.rank == self.dimensional_matrix.rank:
                            if not group4.included_within(repeating_variables):
                                repeating_variables.append(group4)
    if self.dimensional_matrix.rank == 5:
        for parameter1 in self.parameters[1:]:
            for parameter2 in self.parameters[1:]:
                for parameter3 in self.parameters[1:]:
                    for parameter4 in self.parameters[1:]:
                        for parameter5 in self.parameters[1:]:
                            group5 = ListOfParameters([parameter1, parameter2, parameter3, parameter4, parameter5])
                            M = DimensionalMatrix(group5)
                            if M.rank == self.dimensional_matrix.rank:
                                if not group5.included_within(repeating_variables):
                                    repeating_variables.append(group5)
    print('Length of repeating variable list', len(repeating_variables))
    for repeating_variable in repeating_variables:
        print(repeating_variable.units)
    return repeating_variables
