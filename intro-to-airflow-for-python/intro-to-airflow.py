from airflow.models import DAG


def a_simple_dag():
    # Define the default_args dictionary
    default_args = {
        'owner': 'dsmith',
        'start_date': datetime(2020, 1, 14),
        'retries': 2
    }

    # Instantiate the DAG object
    etl_dag = DAG('example_etl', default_args=default_args)


def main():
    a_simple_dag()


if __name__ == '__main__':
    main()
