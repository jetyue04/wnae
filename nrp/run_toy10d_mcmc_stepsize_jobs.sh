#!/bin/bash

# --- CONFIG ---
STEP_SIZES=(0.1 0.5 1 4)
N_STEPS=5000

for step in "${STEP_SIZES[@]}"; do
  JOB_NAME="wnae-job-toy10d-unstandardized-mcmc${step//./}"
  OUTPUT_DIR="output/toy10d-unstandardized-mcmc${step//./}"

  cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: ${JOB_NAME}
  namespace: axol1tl
spec:
  template:
    spec:
      restartPolicy: Never
      containers:
        - name: tuner
          image: python:3.10-slim
          imagePullPolicy: Always
          volumeMounts:
            - name: axovol
              mountPath: /axovol
          workingDir: /axovol/wnae_10_14
          command:
            - "/bin/bash"
            - "-c"
            - |
              set -e
              python -m pip install --no-cache-dir -r requirements.txt
              python -m pip install numpy==1.23.5 scipy==1.13.1
              python -u train_toy.py --override \
                data.N=3 \
                data.D=10 \
                training.n_epochs=500 \
                data.output="${OUTPUT_DIR}" \
                wnae.x_step_size=${step} \
                | tee train_${JOB_NAME}.log
          resources:
            requests:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 2
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: 2
      volumes:
        - name: axovol
          persistentVolumeClaim:
            claimName: axovol
EOF

done
