"""Run the frozen worker with gpu025 enabled for the recovery campaign."""
import run_worker


if __name__ == '__main__':
    run_worker.ALLOWED_NODES = (*run_worker.ALLOWED_NODES, 'gpu025')
    run_worker.main()
