import os
from dotenv import load_dotenv
import time
import os
import docker
import logging
import requests
from torch.cuda import device_count


# Get the path to the directory this file is in
BASEDIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
load_dotenv(os.path.join(BASEDIR, ".env"))

logger = logging.getLogger(__name__)

class TGIServer:
    def __init__(self, model_name, port=8080, endpoint="http://localhost",
                 max_batch_total_tokens=32000, max_concurrent_requests=500):
        # does 3710 need to be hardcoded?
        os.environ["DOCKER_HOST"] = "unix:///run/user/3710/docker.sock"

        self.dockerclient = docker.from_env()

        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", None)

        if visible_devices is not None:
            # use visible devices
            self.device = visible_devices
        else:
            # torch get visible devices
            n_devices = device_count()
            self.device = ",".join([str(x) for x in range(n_devices)])

        logger.info(f"Using device(s): {self.device}")

        self.model = model_name
        self.volume = os.getenv("TGI_VOLUME")
        self.token = os.getenv("HF_READ_TOKEN")
        self.endpoint = endpoint
        self.port = port
        self.health_endpoint = f"{endpoint}:{port}/health"
        self.tgi_container = "ghcr.io/huggingface/text-generation-inference:latest"

        self.max_batch_total_tokens = max_batch_total_tokens
        self.max_concurrent_requests = max_concurrent_requests

        self.startup_wait_time = 30
        self.startup_iterations = 50

    def _is_healthy(self):
        response = requests.get(self.health_endpoint)

        if response.status_code != 200:
            return False
        else:
            return True

    def is_healthy(self):
        try:
            return self._is_healthy()
        except Exception as e:
            logger.error(e)
            return False
        
    def read_logs(self):
        return self.container.logs(tail=10)

    def _wait_for_startup(self):
        errors = []
        logger.info("Waiting for container to start")

        for i in range(self.startup_iterations):
            try:
                server_health = self._is_healthy()
            except Exception as e:
                errors.append(e)
                server_health = False

            if server_health:
                logger.info("Container is healthy")
                return True
            else:
                logger.info("Container is not healthy")
                log_tail = self.read_logs()
                logger.info(f"Container logs:")
                logger.info(f'{"-"*20}')
                logger.info(log_tail.decode("utf-8"))
                logger.info(f'{"-"*20}')
                logger.info(
                    f"{i}/{self.startup_iterations} Waiting {self.startup_wait_time} seconds"
                )
                time.sleep(self.startup_wait_time)

        logger.info("Container failed to start")
        if errors:
            logger.info("Errors during startup:")
            for error in errors:
                logger.info(error)

        return False


    def search_for_container(self, model_name, port):
        def get_model_id(c: docker.models.containers.Container):
            attr_args = c.attrs['Args']
            # iterate through attr_args two at a time
            for i in range(0, len(attr_args), 2):
                if attr_args[i] == "--model-id":
                    return attr_args[i+1]
            
            return None
        
        def get_host_ports(c: docker.models.containers.Container):
            ports = set()
            for portkey, hostinfo_list in c.ports.items():
                    for hostinfo in hostinfo_list:
                        for info, port in hostinfo.items():
                            if info == 'HostPort':
                                ports.add(port)
            return ports

        containers = self.dockerclient.containers.list()
        for container in containers:
            model_id = get_model_id(container)
            if model_id == model_name:
                host_ports = get_host_ports(container)
                if port in host_ports:
                    return container
                
        return None


    def start(self):
        logger.info("Starting container")
        # check if container is already running
        server_healthy = self.is_healthy()

        if server_healthy:
            container = self.search_for_container(self.model, self.port)
            if container:
                self.container = container
                return True
            else:
                logger.info(f"Found container at address: {self.endpoint}:{self.port}, but model id does not match {self.model}")
                return False

        
        if not server_healthy:        
            self.container = self.dockerclient.containers.run(
                self.tgi_container,
                f"--model-id {self.model}",
                shm_size="1g",
                device_requests=[
                    docker.types.DeviceRequest(
                        device_ids=[self.device], capabilities=[["gpu"]]
                    )
                ],
                ports={"80": self.port},
                environment={
                    "HUGGING_FACE_HUB_TOKEN": self.token,
                    "MAX_BATCH_TOTAL_TOKENS": self.max_batch_total_tokens,
                    "MAX_CONCURRENT_REQUESTS": self.max_concurrent_requests,},
                volumes=[f"{self.volume}:/data"],
                detach=True,
                publish_all_ports=False,
                network_mode="default",
                # dtype="bfloat16",
            )

        if not self._wait_for_startup():
            raise Exception("Container failed to start")
        else:
            return True

    def get(self, container_name):
        logger.info(f"Getting container: {container_name}")
        self.container = self.dockerclient.containers.get(container_name)
        try:
            server_health = self.is_healthy()
        except Exception as e:
            # log error
            logger.error(e)
            server_health = False
        
        if not server_health:
            logger.error(f"Container {container_name} is not healthy")
            return False
    
        return True


    def stop(self):
        logger.info("Stopping container")
        self.container.stop()
