import os.path
import shutil

from pyiron_atomistics._tests import TestWithProject


class TestVaspRestart(TestWithProject):
    """
    Tests invariants of VaspBase.restart()
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        static_path = os.path.join(
                cls.file_location, "../static/vasp_test_files/"
        )

        # manually fix up a job that "aborted"
        # since it's an aborted job we cannot import it the same way we do the
        # other examples, so instead:
        #   1. create a job and save it to the database and working directory
        #   2. overwrite its files
        #   3. manually change the job status to aborted
        aborted_path = os.path.join(static_path, "full_job_aborted")
        cls.aborted_job = cls.project.create.job.Vasp("aborted")
        cls.aborted_job.structure = cls.project.create.structure.read(
                os.path.join(aborted_path, "POSCAR")
        )
        # Normally we would do this
        # cls.aborted_job.save(
        # but because we cannot ship Vasp POTCARs Vasp.write_input() won't work in the test env
        db_entry = cls.aborted_job.db_entry()
        db_entry["status"] = "aborted"
        job_id = cls.project.db.add_item_dict(db_entry)
        cls.aborted_job.reset_job_id(job_id)
        cls.aborted_job.to_hdf()
        cls.aborted_job._create_working_directory()

        shutil.copytree(aborted_path, cls.aborted_job.working_directory, dirs_exist_ok=True)

        # Now simulate a job that timed out on a cluster queue
        cls.project.import_from_path(
                path=os.path.join(static_path, "full_job_sample"),
                recursive=False, copy_raw_files=True
        )
        cls.timeout_job = cls.project.load("full_job_sample")
        cls.timeout_job.status.collect = True
        cls.timeout_job.decompress()

    def test_collect_on_restart(self):
        """Calling restart on non-finished job that can be collected should do so."""
        self.timeout_job.restart()
        # since we decompress the job on setup, we can use it as a cannary that restart called collect and compress
        # again
        self.assertTrue(self.timeout_job.is_compressed(), "Job not recollected after restart!")

    def test_restart_aborted(self):
        """Calling restart on an aborted job that cannot be collected should not error."""
        try:
            self.aborted_job.restart()
        except:
            self.fail("Restart raised an error!")

if __name__ == "__main__":
    unittest.main()
