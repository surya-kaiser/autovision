#!/bin/bash
# Fix docker socket permissions at startup so jenkins user can access it
if [ -S /var/run/docker.sock ]; then
    DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
    # Create/update docker group with host GID
    if ! getent group docker > /dev/null 2>&1; then
        groupadd -g "${DOCKER_GID}" docker
    else
        groupmod -g "${DOCKER_GID}" docker
    fi
    usermod -aG docker jenkins
    chmod 666 /var/run/docker.sock
    echo "Docker socket fixed: GID=${DOCKER_GID}"
fi

# Drop back to jenkins user and run Jenkins
exec gosu jenkins /usr/bin/tini -- /usr/local/bin/jenkins.sh "$@"
