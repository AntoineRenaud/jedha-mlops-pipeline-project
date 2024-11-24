# Sample Jenkins server 

* Clone the repository
* `cd sample-jenkins-server`
* `docker compose up`

## Connect to the server

* `http://localhost:8080`
* You will need to provide a password: `docker exec -it sample-jenkins-server cat /var/jenkins_home/secrets/initialAdminPassword`


