
class YaffSeed:
    """Simple datastructure to represent a seed for YAFF force fields

    Seeds contain all the data and parameters necessary to construct a force
    field unambiguously.

    """

    def __init__(self, system, parameters, ff_args):
        """Constructor

        Parameters
        ----------

        system : yaff.System
            yaff system necessary to apply the generators

        parameters : yaff.Parameters
            contains actual force field parameters

        ff_args : yaff.FFArgs
            contains meta parameters such as the cutoff radius for nonbonded
            interactions

        """
        self.system = system
        self.parameters = parameters
        self.ff_args = ff_args


class OpenMMSeed:
    """Simple datastructure to represent a seed for OpenMM force fields

    Seeds contain all the data and parameters necessary to construct a force
    field unambiguously.

    """

    def __init__(self, system_mm):
        """Constructor

        Parameters
        ----------

        system_mm : mm.System
            contains the particle properties and all forces present in the
            system.

        """
        self.system_mm = system_mm
