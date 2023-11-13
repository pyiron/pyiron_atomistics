from typing import Union, List
from mp_api.client import MPRester
from pyiron_atomistics.atomistics.structure.has_structure import HasStructure
from pyiron_atomistics.atomistics.structure.structurestorage import StructureStorage
from pyiron_atomistics.atomistics.structure.atoms import pymatgen_to_pyiron, Atoms


class MPQueryResults(HasStructure):
    def __init__(self, results):
        """

        Args:
            docs (list of dicts): query results from Materials Project, should be obtained with use_document_model=False
        """
        self._results = results
        self._material_ids = [r["material_id"] for r in results]

    def _translate_frame(self, frame):
        try:
            return self._material_ids.index(frame)
        except ValueError:
            raise KeyError(f"material id {frame} not among results!") from None

    def __iter__(self):
        yield from self.iter_structures()

    def _get_structure(self, frame, wrap_atoms=True):
        return pymatgen_to_pyiron(self._results[frame]["structure"])

    def _number_of_structures(self):
        return len(self._results)

    def to_list(self):
        """
        Get a list of queried structures.

        Returns:
            list: structures
        """
        return [pymatgen_to_pyiron(r["structure"]) for r in self._results]

    def to_storage(self):
        """
        Get a StructureStorage of queried structures.

        The materials project id is used as the identifier.

        Returns:
            :class:`~.StructureStorage`: structures
        """
        store = StructureStorage()
        for i, structure in enumerate(self):
            store.add_structure(structure, identifier=self._material_ids[i])
        return store


class MaterialsProjectFactory:
    """
    Convenience interface to the Materials Project Structure Database.

    Usage is only possible with an API key obtained from the Materials Project.  To do this, create an account with
    them, login and access `this webpage <https://next-gen.materialsproject.org/api#api-key>`.

    Once you have a key, either pass it as the `api_key` parameter in the methods of this object or export an
    environment variable, called `MP_API_KEY`, in your shell setup.
    """

    @staticmethod
    def search(
        chemsys: Union[str, List[str]], api_key=None, **kwargs
    ) -> MPQueryResults:
        """
        Search the database for all structures matching the given query.

        Note that `chemsys` takes distint values for unaries, binaries and so!  A query with `chemsys=["Fe", "O"]` will
        return iron structures and oxygen structures, but no iron oxide structures.  Similarily `chemsys=["Fe-O"]` will
        not return unary structures.

        All keyword arguments for filtering from the original API are supported.  See the
        `original docs <https://docs.materialsproject.org/downloading-data/using-the-api>`_ for them.

        Search for all iron structures:

        >>> pr = Project(...)
        >>> irons = pr.create.structure.materialsproject.search("Fe")
        >>> irons.number_of_structures
        10

        The returned :class:`~.MPQueryResults` object implements :class:`~.HasStructure` and can be accessed with the
        material ids as a short-hand

        >>> irons.get_structure(1) == irons.get_structure('mp-13')
        True

        Search for all structures with Al, Li that are on the T=0 convex hull:

        >>> alli = pr.create.structure.materialsproject.search(['Al', 'Li', 'Al-Li'], is_stable=True)
        >>> len(alli)
        6

        Args:
            chemsys (str, list of str): confine search to given elements; either an element symbol or multiple element
            symbols seperated by dashes; if a list of strings is given return structures matching either of them
            api_key (str, optional): if your API key is not exported in the environment flag MP_API_KEY, pass it here
            **kwargs: passed verbatim to :meth:`mp_api.MPRester.summary.search` to further filter the results

        Returns:
            :class:`~.MPQueryResults`: resulting structures from the query
        """
        rest_kwargs = {
            "use_document_model": False,  # returns results as dictionaries
            "include_user_agent": True,  # send some additional software version info to MP
        }
        if api_key is not None:
            rest_kwargs["api_key"] = api_key
        with MPRester(**rest_kwargs) as mpr:
            results = mpr.summary.search(
                chemsys=chemsys, **kwargs, fields=["structure", "material_id"]
            )
        return MPQueryResults(results)

    @staticmethod
    def by_id(
        material_id: Union[str, int],
        final: bool = True,
        conventional_unit_cell: bool = False,
        api_key=None,
    ) -> Union[Atoms, List[Atoms]]:
        """
        Retrieve a structure by material id.

        This is how you would ask for the iron ground state:

        >>> pr = Project(...)
        >>> pr.create.structure.materialsproject.by_id('mp-13')
        Fe: [0. 0. 0.]
        tags:
            spin: [(0: 2.214)]
        pbc: [ True  True  True]
        cell:
        Cell([[2.318956, 0.000185, -0.819712], [-1.159251, 2.008215, -0.819524], [2.5e-05, 0.000273, 2.459206]])


        Args:
            material_id (str): the id assigned to a structure by the materials project
            api_key (str, optional): if your API key is not exported in the environment flag MP_API_KEY, pass it here
            final (bool, optional): if set to False, returns the list of initial structures,
            else returns the final structure. (Default is True)
            conventional_unit_cell (bool, optional): if set to True, returns the standard conventional unit cell.
            (Default is False)

        Returns:
            :class:`~.Atoms`: requested final structure if final is True
            list of :class:~.Atoms`:  a list of initial (pre-relaxation) structures if final is False

        Raises:
            ValueError: material id does not exist
        """
        rest_kwargs = {
            "include_user_agent": True,  # send some additional software version info to MP
        }
        if api_key is not None:
            rest_kwargs["api_key"] = api_key
        with MPRester(**rest_kwargs) as mpr:
            if final:
                return pymatgen_to_pyiron(
                    mpr.get_structure_by_material_id(
                        material_id=material_id,
                        final=final,
                        conventional_unit_cell=conventional_unit_cell,
                    )
                )
            else:
                return [
                    pymatgen_to_pyiron(mpr_structure)
                    for mpr_structure in (
                        mpr.get_structure_by_material_id(
                            material_id=material_id,
                            final=final,
                            conventional_unit_cell=conventional_unit_cell,
                        )
                    )
                ]
