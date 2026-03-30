export async function apiGet(baseUrl, path) {
  let response;
  try {
    response = await fetch(`${baseUrl}${path}`);
  } catch (error) {
    throw new Error(
      `Network error reaching ${baseUrl}. Make sure the FastAPI server is running and reachable from the frontend.`,
    );
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}

export async function apiPost(baseUrl, path, payload) {
  let response;
  try {
    response = await fetch(`${baseUrl}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch (error) {
    throw new Error(
      `Network error reaching ${baseUrl}. Make sure the FastAPI server is running and reachable from the frontend.`,
    );
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `${response.status} ${response.statusText}`);
  }
  return response.json();
}
